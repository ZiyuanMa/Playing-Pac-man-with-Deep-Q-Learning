from torch.optim import Adam
import math
import os
import random
import time
from collections import deque
from copy import deepcopy
from model import Network, transform

import gym

import numpy as np
import torch

from buffer import ReplayBuffer, PrioritizedReplayBuffer
import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def learn(  env_name=config.env_name, number_timesteps=config.training_steps,
            save_path='./models/MsPacman', save_interval=config.save_interval,
            gamma=config.gamma, grad_norm=config.grad_norm, double_q=config.double_q,
            dueling=config.dueling, exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps, batch_size=config.batch_size, train_freq=config.train_freq,
            learning_starts=config.learning_starts, target_network_update_freq=config.target_network_update_freq, buffer_size=config.buffer_size,
            prioritized_replay=config.prioritized_replay, prioritized_replay_alpha=config.prioritized_replay_alpha,
            prioritized_replay_beta0=config.prioritized_replay_beta0, atom_num=config.atom_num, min_value=config.min_value, max_value=config.max_value):

    # create environment
    env = gym.make(env_name)
    
    # create network and optimizer
    network = Network(env.action_space.n)
    optimizer = Adam(network.parameters(), 6.25e-5, eps=1.5e-4)

    # create target network
    qnet = network.to(device)
    tar_qnet = deepcopy(qnet)

    # prioritized replay
    if prioritized_replay:
        buffer = PrioritizedReplayBuffer(buffer_size, device,
                                         prioritized_replay_alpha,
                                         prioritized_replay_beta0)
    else:
        buffer = ReplayBuffer(buffer_size, device)

    generator = _generate(device, env, qnet,
                        number_timesteps,
                        exploration_fraction, exploration_final_eps,
                        atom_num, min_value, max_value)

    if atom_num > 1:
        delta_z = float(max_value - min_value) / (atom_num - 1)
        support = torch.linspace(min_value, max_value, atom_num).to(device)
        batch_support = support.unsqueeze(0).expand(batch_size, atom_num)

    start_ts = time.time()
    for n_iter in range(1, number_timesteps + 1):

        if prioritized_replay:
            buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
            
        data = generator.__next__()
        buffer.add(data)


        # update qnet
        if n_iter > learning_starts and n_iter % train_freq == 0:
            batch_obs, batch_action, batch_reward, batch_obs_, batch_done, batch_steps, *extra = buffer.sample(batch_size)

            if atom_num == 1:
                with torch.no_grad():
                    
                    # choose max q index from next observation
                    if double_q:
                        batch_action_ = qnet(batch_obs_).argmax(1).unsqueeze(1)
                        batch_q_ = (1 - batch_done) * tar_qnet(batch_obs_).gather(1, batch_action_)
                    else:
                        batch_q_ = (1 - batch_done) * tar_qnet(batch_obs_).max(1, keepdim=True)[0]

                batch_q = qnet(batch_obs).gather(1, batch_action)

                abs_td_error = (batch_q - (batch_reward + gamma**batch_steps * batch_q_)).abs()

                priorities = abs_td_error.detach().cpu().clamp(1e-6).numpy()

                if extra:
                    loss = (extra[0] * huber_loss(abs_td_error)).mean()
                else:
                    loss = huber_loss(abs_td_error).mean()

            else:
                with torch.no_grad():

                    batch_dist_ = tar_qnet(batch_obs_).exp()
                    if double_q:
                        batch_action_ = (qnet(batch_obs_).exp() * support).sum(-1).argmax(1)
                    else:
                        batch_action_ = (batch_dist_ * support).sum(-1).argmax(1)

                    b_Tz = (gamma**batch_steps * (1 - batch_done) * batch_support + batch_reward).clamp(min_value, max_value)
                    b_i = (b_Tz - min_value) / delta_z
                    b_l = b_i.floor()
                    b_u = b_i.ceil()
                    b_m = torch.zeros(batch_size, atom_num).to(device)
                    temp = batch_dist_[torch.arange(batch_size), batch_action_, :]
                    b_m.scatter_add_(1, b_l.long(), temp * (b_u - b_i))
                    b_m.scatter_add_(1, b_u.long(), temp * (b_i - b_l))

                batch_log_dist = qnet(batch_obs)[torch.arange(batch_size), batch_action.squeeze(1), :]
                kl_error = -(batch_log_dist * b_m).sum(1)

                # use kl error as priorities as proposed by Rainbow
                priorities = kl_error.detach().cpu().clamp(1e-6).numpy()
                loss = kl_error.mean()

            optimizer.zero_grad()
            loss.backward()
            if grad_norm is not None:
                nn.utils.clip_grad_norm_(qnet.parameters(), grad_norm)
            optimizer.step()

            if prioritized_replay:
                buffer.update_priorities(extra[1], priorities)

        # update target net and log
        if n_iter % target_network_update_freq == 0:
            tar_qnet.load_state_dict(qnet.state_dict())
            print('{} Iter {} {}'.format('=' * 10, n_iter, '=' * 10))

            fps = int(target_network_update_freq / (time.time() - start_ts))
            start_ts = time.time()

            print('FPS {}'.format(fps))
            
            if n_iter > learning_starts and n_iter % train_freq == 0:

                print('vloss: {:.6f}'.format(loss.item()))

        if save_interval and n_iter % save_interval == 0:
            torch.save(qnet.state_dict(), os.path.join(save_path, '{}checkpoint++.pth'.format(n_iter)))


def _generate(device, env, qnet,
              number_timesteps,
              exploration_fraction, exploration_final_eps,
              atom_num, min_value, max_value):

    """ Generate training batch sample """

    explore_steps = number_timesteps * exploration_fraction

    if atom_num > 1:
        vrange = torch.linspace(min_value, max_value, atom_num).to(device)

    o = env.reset()
    o = transform(o)
    o = np.concatenate([np.copy(o) for _ in range(4)], axis=0)
    

    for n in range(1, number_timesteps + 1):
        epsilon = 1.0 - (1.0 - exploration_final_eps) * n / explore_steps
        epsilon = max(exploration_final_eps, epsilon)

        # sample action
        with torch.no_grad():

            ob = torch.from_numpy(o).unsqueeze(0).to(device)

            q = qnet(ob)

            if atom_num > 1:
                q = (q.exp() * vrange).sum(2)
            
            # choose action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = q.argmax(1).cpu().numpy()[0]

        # take action in env
        o_, reward, done, _ = env.step(action)
        o_ = transform(o_)

        temp = np.delete(o, 0, 0)
        o_ = np.concatenate((temp, o_), axis=0)
        # return data and update observation

        if reward > 1:
            reward = 1
        elif reward < -1:
            reward = -1

        yield (o, action, reward, o_, int(done))

        if not done:

            o = o_ 
        else:
            o = env.reset()
            o = transform(o)
            o = np.concatenate([np.copy(o) for _ in range(4)], axis=0)


def huber_loss(abs_td_error):
    flag = (abs_td_error < 1).float()
    return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)


if __name__ == '__main__':

    learn()
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim import Adam
import math
import os
import random
import time
from collections import deque
from copy import deepcopy
from torchvision import transforms

import gym

import numpy as np
import torch
import torch.distributions
from torch.nn.functional import softmax, log_softmax

from buffer import ReplayBuffer, PrioritizedReplayBuffer
import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Lambda(lambda x: x[:195,:,:]),
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255),
    transforms.Lambda(lambda x: x.numpy()),
])

class Network(nn.Module):
    def __init__(self, in_shape, out_dim, atom_num, dueling):
        super().__init__()
        c, h, w = in_shape
        cnn_out_dim = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.atom_num = atom_num

        # 84 x 84 input
        self.feature = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            Flatten(),
        )

        self.q = nn.Sequential(
            nn.Linear(cnn_out_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, out_dim * atom_num)
        )
        if dueling:
            self.state = nn.Sequential(
                nn.Linear(cnn_out_dim, 256),
                nn.ReLU(True),
                nn.Linear(256, atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        latent = self.feature(x)
        qvalue = self.q(latent)
        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            qvalue = qvalue.view(batch_size, -1, self.atom_num)
            if hasattr(self, 'state'):
                svalue = self.state(latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs



def learn(  env, number_timesteps,
            save_path='./models/MsPacman/', save_interval=config.save_interval,
            gamma=config.gamma, grad_norm=config.grad_norm, double_q=config.double_q,
            param_noise=config.param_noise, dueling=config.dueling, exploration_fraction=config.exploration_fraction,
            exploration_final_eps=config.exploration_final_eps, batch_size=config.batch_size, train_freq=config.train_freq,
            learning_starts=config.learning_starts, target_network_update_freq=config.target_network_update_freq, buffer_size=config.buffer_size,
            prioritized_replay=config.prioritized_replay, prioritized_replay_alpha=config.prioritized_replay_alpha,
            prioritized_replay_beta0=config.prioritized_replay_beta0, atom_num=config.atom_num, min_value=config.min_value, max_value=config.max_value):
    """

    Parameters:
    ----------
    double_q (bool): if True double DQN will be used
    param_noise (bool): whether or not to use parameter space noise
    dueling (bool): if True dueling value estimation will be used
    exploration_fraction (float): fraction of entire training period over which
                                  the exploration rate is annealed
    exploration_final_eps (float): final value of random action probability
    batch_size (int): size of a batched sampled from replay buffer for training
    train_freq (int): update the model every `train_freq` steps
    learning_starts (int): how many steps of the model to collect transitions
                           for before learning starts
    target_network_update_freq (int): update the target network every
                                      `target_network_update_freq` steps
    buffer_size (int): size of the replay buffer
    prioritized_replay (bool): if True prioritized replay buffer will be used.
    prioritized_replay_alpha (float): alpha parameter for prioritized replay
    prioritized_replay_beta0 (float): beta parameter for prioritized replay
    atom_num (int): atom number in distributional RL for atom_num > 1
    min_value (float): min value in distributional RL
    max_value (float): max value in distributional RL

    """
    # create network and optimizer
    network = Network(config.input_shape, env.action_space.n, atom_num, dueling)
    optimizer = Adam(network.parameters(), 1e-4, eps=1e-5)

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
        z_i = torch.linspace(min_value, max_value, atom_num).to(device)

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
                    batch_doneist_ = tar_qnet(batch_obs_).exp()
                    batch_action_ = (batch_doneist_ * z_i).sum(-1).argmax(1)
                    b_tzj = (gamma * (1 - batch_done) * z_i[None, :] + batch_reward).clamp(min_value, max_value)
                    b_i = (b_tzj - min_value) / delta_z
                    b_l = b_i.floor()
                    b_u = b_i.ceil()
                    b_m = torch.zeros(batch_size, atom_num).to(device)
                    temp = batch_doneist_[torch.arange(batch_size), batch_action_, :]
                    b_m.scatter_add_(1, b_l.long(), temp * (b_u - b_i))
                    b_m.scatter_add_(1, b_u.long(), temp * (b_i - b_l))
                batch_q = qnet(batch_obs)[torch.arange(batch_size), batch_action.squeeze(1), :]
                kl_error = -(batch_q * b_m).sum(1)
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
            torch.save(qnet.state_dict(), os.path.join(save_path, '{}checkpoint.pth'.format(n_iter)))


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


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


if __name__ == '__main__':
    env = gym.make('MsPacman-v0')
    learn(env, 10000000)
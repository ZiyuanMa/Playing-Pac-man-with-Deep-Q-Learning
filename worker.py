'''Replay buffer, learner and actor'''
import time
import random
import os
from collections import deque
from copy import deepcopy
from typing import List, Tuple
import threading
import ray
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
import numpy as np
from model import Network
from environment import creat_env
import config


############################## Replay Buffer ##############################

class SumTree:
    '''store priority for prioritized experience replay''' 
    def __init__(self, capacity: int, tree_dtype=np.float64):
        self.capacity = capacity

        self.layer = 1
        while capacity > 1:
            self.layer += 1
            capacity //= 2
        assert capacity == 1, 'capacity only allow n to the power of 2 size'

        self.tree_dtype = tree_dtype  # float32 is not enough
        self.tree = np.zeros(2**self.layer-1, dtype=self.tree_dtype)

    def sum(self):
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])
        return self.tree[0]

    def __getitem__(self, idx: int):
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity-1+idx]

    def batch_sample(self, batch_size: int):
        p_sum = self.tree[0]
        interval = p_sum/batch_size

        prefixsums = np.arange(0, p_sum, interval, dtype=self.tree_dtype) + np.random.uniform(0, interval, batch_size)

        idxes = np.zeros(batch_size, dtype=np.int)
        for _ in range(self.layer-1):
            nodes = self.tree[idxes*2+1]
            idxes = np.where(prefixsums<nodes, idxes*2+1, idxes*2+2)
            prefixsums = np.where(idxes%2==0, prefixsums-self.tree[idxes-1], prefixsums)
        
        priorities = self.tree[idxes]
        idxes -= self.capacity-1

        assert np.all(priorities>0), 'idx: {}, priority: {}'.format(idxes, priorities)
        assert np.all(idxes>=0) and np.all(idxes<self.capacity)

        return idxes, priorities

    def batch_update(self, idxes: np.ndarray, priorities: np.ndarray):
        idxes += self.capacity-1
        self.tree[idxes] = priorities

        for _ in range(self.layer-1):
            idxes = (idxes-1) // 2
            idxes = np.unique(idxes)
            self.tree[idxes] = self.tree[2*idxes+1] + self.tree[2*idxes+2]
        
        # check
        assert np.sum(self.tree[-self.capacity:])-self.tree[0] < 0.1, 'sum is {} but root is {}'.format(np.sum(self.tree[-self.capacity:]), self.tree[0])

@ray.remote(num_cpus=1)
class ReplayBuffer:
    def __init__(self, buffer_capacity=config.buffer_capacity, slot_capacity=config.slot_capacity,
                alpha=config.prioritized_replay_alpha, beta=config.prioritized_replay_beta0,
                batch_size=config.batch_size, frame_stack=config.frame_stack):

        self.buffer_capacity = buffer_capacity
        self.slot_capacity = slot_capacity
        self.num_slots = buffer_capacity//slot_capacity
        self.slot_ptr = 0

        # prioritized experience replay
        self.priority_tree = SumTree(buffer_capacity)
        self.alpha = alpha
        self.beta = beta

        self.batched_data = []
        self.batch_size = batch_size
        self.frame_stack = frame_stack

        self.lock = threading.Lock()

        self.obs_buf = [None for _ in range(self.num_slots)]
        self.act_buf = [None for _ in range(self.num_slots)]
        self.rew_buf = [None for _ in range(self.num_slots)]
        self.gamma_buf = [None for _ in range(self.num_slots)]
        self.size_buf = np.zeros(self.num_slots, dtype=np.uint)

    def __len__(self):
        return np.sum(self.size_buf).item()
    
    def run(self):
        self.background_thread = threading.Thread(target=self.prepare_data, daemon=True)
        self.background_thread.start()

    def prepare_data(self):
        '''background thread'''
        while True:
            if len(self.batched_data) < 4:
                data = self.sample_batch(self.batch_size)
                data_id = ray.put(data)
                self.batched_data.append(data_id)
            else:
                time.sleep(0.1)

    def get_data(self):
        '''call by learner to get batched data'''

        if len(self.batched_data) == 0:
            print('no batched data')
            data = self.sample_batch(self.batch_size)
            data_id = ray.put(data)
            return data_id
        else:
            return self.batched_data.pop(0)

    def add(self, data_list: List[tuple]):
        '''Call by actors to add data to replaybuffer

        Args:
            data_list: tuples of data, each tuple represents a slot
                in each tuple: 0 obs, 1 action, 2 reward, 3 gamma, 4 td_errors
        '''

        with self.lock:
            for data in data_list:
                idxes = np.arange(self.slot_ptr*self.slot_capacity, (self.slot_ptr+1)*self.slot_capacity)

                slot_size = np.size(data[0], 0) - 4
                self.size_buf[self.slot_ptr] = slot_size

                self.priority_tree.batch_update(idxes, data[4]**self.alpha)

                self.obs_buf[self.slot_ptr] = torch.from_numpy(data[0].astype(np.float16))
                self.act_buf[self.slot_ptr] = torch.from_numpy(data[1].astype(np.long))
                self.rew_buf[self.slot_ptr] = torch.from_numpy(data[2])
                self.gamma_buf[self.slot_ptr] = torch.from_numpy(data[3])

                self.slot_ptr = (self.slot_ptr+1) % self.num_slots

    def sample_batch(self, batch_size):
        '''sample one batch of training data'''
        batch_obs, batch_action, batch_reward, batch_next_obs, batch_gamma = [], [], [], [], []
        idxes, priorities = [], []
        
        with self.lock:

            idxes, priorities = self.priority_tree.batch_sample(self.batch_size)
            global_idxes = idxes // self.slot_capacity
            local_idxes = idxes % self.slot_capacity

            for global_idx, local_idx in zip(global_idxes.tolist(), local_idxes.tolist()):
                
                assert local_idx < len(self.act_buf[global_idx]), 'index is {} but size is {}'.format(local_idx, len(self.act_buf[global_idx]))

                forward_steps = min(config.forward_steps, self.size_buf[global_idx].item()-local_idx)
                obs_sequence = self.obs_buf[global_idx][local_idx:local_idx+4+forward_steps]  # it includes obs and next_obs
                
                obs = obs_sequence[:self.frame_stack]
                action = self.act_buf[global_idx][local_idx]
                reward = self.rew_buf[global_idx][local_idx]
                gamma = self.gamma_buf[global_idx][local_idx]
                next_obs = obs_sequence[-self.frame_stack:]
                
                batch_obs.append(obs)
                batch_action.append(action)
                batch_reward.append(reward)
                batch_next_obs.append(next_obs)
                batch_gamma.append(gamma)

            # importance sampling weight
            min_p = np.min(priorities)
            weights = np.power(priorities/min_p, -self.beta)

            data = (
                torch.stack(batch_obs),
                torch.stack(batch_action).unsqueeze(1),
                torch.stack(batch_reward).unsqueeze(1),
                torch.stack(batch_next_obs),
                torch.stack(batch_gamma).unsqueeze(1),

                idxes,
                torch.from_numpy(weights.astype(np.float16)).unsqueeze(1),
                self.slot_ptr
            )

            return data

    def update_priorities(self, idxes: np.ndarray, priorities: np.ndarray, old_ptr: int):
        """Update priorities of sampled transitions"""
        with self.lock:

            # discard the indices that already been discarded in replay buffer during training
            if self.slot_ptr > old_ptr:
                # range from [old_ptr, self.slot_ptr)
                mask = (idxes < old_ptr*self.slot_capacity) | (idxes >= self.slot_ptr*self.slot_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]
            elif self.slot_ptr < old_ptr:
                # range from [0, self.slot_ptr) & [old_ptr, self,capacity)
                mask = (idxes < old_ptr*self.slot_capacity) & (idxes >= self.slot_ptr*self.slot_capacity)
                idxes = idxes[mask]
                priorities = priorities[mask]

            self.priority_tree.batch_update(np.copy(idxes), np.copy(priorities)**self.alpha)

    def ready(self):
        if len(self) >= config.learning_starts:
            return True
        else:
            return False


############################## Learner ##############################

@ray.remote(num_cpus=1, num_gpus=1)
class Learner:
    def __init__(self, buffer, env_name=config.env_name, lr=config.lr, eps=config.eps):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.env_name = env_name
        self.model = Network(creat_env().action_space.n)
        self.model.to(self.device)
        self.model.train()
        self.tar_model = deepcopy(self.model)
        self.tar_model.eval()
        self.optimizer = Adam(self.model.parameters(), lr=lr, eps=eps)
        self.buffer = buffer
        self.counter = 0
        self.last_counter = 0
        self.done = False
        self.loss = 0

        self.store_weights()

    def get_weights(self):
        return self.weights_id

    def store_weights(self):
        state_dict = self.model.state_dict()
        for k, v in state_dict.items():
            state_dict[k] = v.cpu()
        self.weights_id = ray.put(state_dict)

    def run(self):
        self.learning_thread = threading.Thread(target=self.train, daemon=True)
        self.learning_thread.start()

    def train(self):
        scaler = GradScaler()

        while self.counter < config.training_steps:

            data_id = ray.get(self.buffer.get_data.remote())
            data = ray.get(data_id)

            batch_obs, batch_action, batch_reward, batch_next_obs, batch_gamma, idxes, weights, old_ptr = data
            batch_obs, batch_action, batch_reward= batch_obs.to(self.device), batch_action.to(self.device), batch_reward.to(self.device)
            batch_next_obs, batch_gamma, weights = batch_next_obs.to(self.device), batch_gamma.to(self.device), weights.to(self.device)

            with torch.no_grad():
                # double q learning
                batch_action_ = self.model(batch_next_obs).argmax(1).unsqueeze(1)
                batch_q_ = self.tar_model(batch_next_obs).gather(1, batch_action_)

            batch_q = self.model(batch_obs).gather(1, batch_action)

            td_error = (batch_q - (batch_reward + batch_gamma * batch_q_))

            priorities = td_error.detach().squeeze().abs().clamp(1e-4).cpu().numpy()

            loss = (weights * self.huber_loss(td_error)).mean()
            self.loss += loss.item()

            # automatic mixed precision training
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 40)
            scaler.step(self.optimizer)
            scaler.update()

            self.counter += 1

            self.buffer.update_priorities.remote(idxes, priorities, old_ptr)

            # store new weights in shared memory
            if self.counter % 5  == 0:
                self.store_weights()

            # update target net, save model
            if self.counter % config.target_network_update_freq == 0:
                self.tar_model.load_state_dict(self.model.state_dict())
            
            if self.counter % config.save_interval == 0:
                torch.save(self.model.state_dict(), os.path.join('models', '{}{}.pth'.format(self.env_name, self.counter)))

        self.done = True

    def huber_loss(self, td_error, kappa=1.0):
        abs_td_error = td_error.abs()
        flag = (abs_td_error < kappa).half()
        return flag * abs_td_error.pow(2) * 0.5 + (1 - flag) * (abs_td_error - 0.5)

    def stats(self, interval: int):
        print('number of updates: {}'.format(self.counter))
        print('update speed: {}/s'.format((self.counter-self.last_counter)/interval))
        if self.counter != self.last_counter:
            print('loss: {:.4f}'.format(self.loss / (self.counter-self.last_counter)))
        print('buffer size: {}'.format(ray.get(self.buffer.__len__.remote())))
        
        self.last_counter = self.counter
        self.loss = 0
        return self.done


############################## Actor ##############################

class LocalBuffer:
    '''store transition of one episode'''
    def __init__(self, init_obs, n_step=config.forward_steps, frame_stack=config.frame_stack, slot_capacity=config.slot_capacity):
        self.n_step = n_step
        self.frame_stack = frame_stack
        self.slot_capacity = slot_capacity
        self.obs_buffer = [init_obs for _ in range(frame_stack)]
        self.action_buffer = []
        self.reward_buffer = []
        self.qval_buffer = []
        self.size = 0
    
    def __len__(self):
        return self.size

    def add(self, action: int, reward: float, next_obs: np.ndarray, q_value):
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.obs_buffer.append(next_obs)
        self.qval_buffer.append(q_value)
        self.size += 1
    
    def finish(self, last_q_val=None) -> List[tuple]:
        cumulated_gamma = [config.gamma**self.n_step for _ in range(self.size-self.n_step)]

        if last_q_val:
            self.qval_buffer.append(last_q_val)
            cumulated_gamma.extend([config.gamma**i for i in reversed(range(self.n_step))])
        else:
            self.qval_buffer.append(np.zeros_like(self.qval_buffer[-1]))
            cumulated_gamma.extend([0 for _ in range(self.n_step)]) # set gamma to 0 so don't need 'done'

        cumulated_gamma = np.array(cumulated_gamma, dtype=np.float16)
        self.obs_buffer = np.concatenate(self.obs_buffer)
        self.action_buffer = np.array(self.action_buffer, dtype=np.uint8)
        self.qval_buffer = np.concatenate(self.qval_buffer)
        self.reward_buffer = self.reward_buffer + [0 for _ in range(self.n_step-1)]
        cumulated_reward = np.convolve(self.reward_buffer, [config.gamma**(self.n_step-1-i) for i in range(self.n_step)],'valid').astype(np.float16)

        num_slots = self.size // self.slot_capacity + 1

        # td_errors
        td_errors = np.zeros(num_slots*self.slot_capacity, dtype=np.float32)
        max_qval = np.max(self.qval_buffer[self.n_step:self.size+1], axis=1)
        max_qval = np.concatenate((max_qval, np.array([max_qval[-1] for _ in range(self.n_step-1)])))
        target_qval = self.qval_buffer[np.arange(self.size), self.action_buffer]
        td_errors[:self.size] = np.abs(cumulated_reward+max_qval-target_qval).clip(1e-4)

        slots = []
        for i in range(num_slots):
            obs = self.obs_buffer[i*config.slot_capacity:(i+1)*config.slot_capacity+4]
            actions = self.action_buffer[i*config.slot_capacity:(i+1)*config.slot_capacity]
            rewards = cumulated_reward[i*config.slot_capacity:(i+1)*config.slot_capacity]
            td_error = td_errors[i*config.slot_capacity:(i+1)*config.slot_capacity]
            gamma = cumulated_gamma[i*config.slot_capacity:(i+1)*config.slot_capacity]

            slots.append((obs, actions, rewards, gamma, td_error))
        
        return slots

@ray.remote(num_cpus=1)
class Actor:
    def __init__(self, epsilon, learner, buffer):

        self.env = creat_env()
        self.model = Network(self.env.action_space.n)
        self.model.eval()
        
        self.obs_history = deque([], maxlen=config.frame_stack)
        self.epsilon = epsilon
        self.learner = learner
        self.replay_buffer = buffer
        self.max_episode_length = config.max_episode_length
        self.counter = 0

    def run(self):
        done = False
        local_buffer = self.reset()
        
        while True:
            obs = torch.from_numpy(np.concatenate(self.obs_history).astype(np.float32)).unsqueeze(0)
            action, qval = self.model.step(obs)

            if random.random() < self.epsilon:
                action = self.env.action_space.sample()

            # apply action in env
            next_obs, reward, done, _ = self.env.step(action)

            self.obs_history.append(next_obs)

            local_buffer.add(action, reward, next_obs, qval)

            if done or len(local_buffer) == self.max_episode_length:
                # finish and send buffer
                if done:
                    data = local_buffer.finish()
                else:
                    obs = torch.from_numpy(np.concatenate(self.obs_history).astype(np.float32)).unsqueeze(0)
                    _, q_val = self.model.step(obs)
                    data = local_buffer.finish(q_val)

                self.replay_buffer.add.remote(data)
                done = False
                self.update_weights()
                local_buffer = self.reset()

            self.counter += 1
            # if self.counter == 400:
            #     self.update_weights()
            #     self.counter = 0

    def update_weights(self):
        '''load latest weights from learner'''
        weights_id = ray.get(self.learner.get_weights.remote())
        weights = ray.get(weights_id)
        self.model.load_state_dict(weights)
    
    def reset(self):
        obs = self.env.reset()
        self.obs_history = deque([obs for _ in range(config.frame_stack)], maxlen=config.frame_stack)
        local_buffer = LocalBuffer(obs)
        return local_buffer


import config
from model import Network, transform
import gym
import torch
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
device = torch.device('cpu')



def test(env_name=config.env_name):
    env = gym.make(env_name)
    round = 5
    show = False
    x = [5*i for i in range(1, 201)]

    network = Network(env.action_space.n, dueling=False, atom_num=1)
    network.to(device)
    checkpoint = config.save_interval

    y1 = []
    while os.path.exists('./models/MsPacman/'+str(checkpoint)+'checkpoint+.pth'):
        network.load_state_dict(torch.load('./models/MsPacman/'+str(checkpoint)+'checkpoint+.pth'))
        sum_reward = 0
        for _ in range(round):
            o = env.reset()
            o = transform(o)
            o = np.concatenate([np.copy(o) for _ in range(4)], axis=0)
            done = False
            while not done:
                if show:
                    env.render()
                    time.sleep(0.05)
                obs = torch.from_numpy(o).unsqueeze(0).to(device)

                q = network(obs)

                if random.random() < config.test_epsilon:
                    action = env.action_space.sample()
                else:
                    action = q.argmax(1).item()

                o_, reward, done, _ = env.step(action)
                o_ = transform(o_)
                temp = np.delete(o, 0, 0)
                o = np.concatenate((temp, o_), axis=0)
                sum_reward += reward

            if show:
                env.close()

        print(' checkpoint: {}' .format(checkpoint))
        print(' avg reward: {}' .format(sum_reward/round))
        y1.append(sum_reward/round)
        checkpoint += config.save_interval

    network = Network(env.action_space.n, dueling=True, atom_num=51)
    network.to(device)
    checkpoint = config.save_interval * 180
    show = False

    vrange = torch.linspace(config.min_value, config.max_value, 51).to(device)
    y2 = []
    while os.path.exists('./models/MsPacman/'+str(checkpoint)+'checkpoint++.pth'):
        network.load_state_dict(torch.load('./models/MsPacman/'+str(checkpoint)+'checkpoint++.pth'))
        sum_reward = 0
        for _ in range(round):
            o = env.reset()
            o = transform(o)
            o = np.concatenate([np.copy(o) for _ in range(4)], axis=0)
            done = False
            while not done:
                if show:
                    env.render()
                    time.sleep(0.05)
                obs = torch.from_numpy(o).unsqueeze(0).to(device)

                q = network(obs)

                q = (q.exp() * vrange).sum(2)

                if random.random() < config.test_epsilon:
                    action = env.action_space.sample()
                else:
                    action = q.argmax(1).item()

                o_, reward, done, _ = env.step(action)
                o_ = transform(o_)
                temp = np.delete(o, 0, 0)
                o = np.concatenate((temp, o_), axis=0)
                sum_reward += reward

            if show:
                env.close()

        print(' checkpoint: {}' .format(checkpoint))
        print(' avg reward: {}' .format(sum_reward/round))
        y2.append(sum_reward/round)
        checkpoint += config.save_interval

    print(sum(y1)/len(y1))
    print(sum(y2)/len(y2))
    plt.title(config.env_name)
    plt.xlabel('number of frames (1e4)')
    plt.ylabel('average reward')

    plt.plot(x, y1, label='DQN')
    plt.plot(x, y2, label='Fusion DQN')
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    
    test('MsPacman-v0')

    # env = gym.make('MsPacman-v0')
    # obs = env.reset()
    # plt.imsave('MsPacman.png', obs)



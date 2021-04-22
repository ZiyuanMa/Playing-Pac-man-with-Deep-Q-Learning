import os
import time
import random
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Network
from environment import creat_env
import config
device = torch.device('cpu')


def test(env_name=config.env_name, save_interval = config.save_interval, test_epsilon=config.test_epsilon,
        show=config.render, save_plot=config.save_plot):

    env = creat_env(env_name, noop_start=False, clip_rewards=False, frame_stack=True)
    test_round = 5
    x, y = [], []

    network = Network(env.action_space.n)
    network.to(device)
    checkpoint = save_interval

    while os.path.exists('./models/{}{}.pth'.format(env_name, checkpoint)):
        x.append(checkpoint)
        network.load_state_dict(torch.load('./models/{}{}.pth'.format(env_name, checkpoint)))
        sum_reward = 0
        for _ in range(test_round):
            obs = env.reset()
            done = False
            while not done:
                if show:
                    env.render()
                    time.sleep(0.05)

                obs = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                action, _ = network.step(obs)

                if random.random() < test_epsilon:
                    action = env.action_space.sample()

                next_obs, reward, done, _ = env.step(action)
                obs = next_obs
                sum_reward += reward

            if show: env.close()

        print(' checkpoint: {}' .format(checkpoint))
        print(' average reward: {}\n' .format(sum_reward/test_round))
        y.append(sum_reward/test_round)
        checkpoint += save_interval

    plt.title(env_name)
    plt.xlabel('training steps')
    plt.ylabel('average reward')

    plt.plot(x, y)
    
    if save_plot:
        plt.savefig('./{}.jpg'.format(env_name))
    plt.show()


if __name__ == '__main__':
    
    test()



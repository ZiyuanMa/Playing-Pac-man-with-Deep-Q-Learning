import config
from dqn import Network
import gym
import torch
import os
import time
import numpy as np
import random
from torchvision import transforms
import matplotlib.pyplot as plt
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




def test(env_name):
    env = gym.make(env_name)
    round = 3
    show = False
    x = [5*i for i in range(1, 101)]

    network = Network(config.input_shape, env.action_space.n, 1, False)
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

                if random.random() < 0.05:
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

    network = Network(config.input_shape, env.action_space.n, 51, True)
    network.to(device)
    checkpoint = config.save_interval
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

                if random.random() < 0.05:
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


    plt.title('MsPacman')
    plt.xlabel('number of frames (1e4)')
    plt.ylabel('average reward')
    

    plt.plot(x, y1, label='DQN')
    plt.plot(x, y2, label='Fusion DQN')
    
    plt.show()


if __name__ == '__main__':
    
    # test('MsPacman-v0')

    env = gym.make('MsPacman-v0')
    obs = env.reset()
    plt.imsave('MsPacman.png', obs)
    # env = gym.make('MsPacman-v0')
    # obs = env.reset()
    # plt.imshow(obs)
    # img.show()


import argparse
import config
from dqn import Network
import gym
import torch
import os
import time
import random
from torchvision import transforms
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# parser.add_argument('--algorithm', type=str, help='Algorithm')
parser.add_argument('--env_name', type=str, default='MsPacman-v0', help='environment name')
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((44, 44)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x / 255),
    transforms.Lambda(lambda x: x.to(device)),
])

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((44, 44)),

])


def test(env_name):
    env = gym.make(env_name)
    network = Network(config.input_shape, env.action_space.n, config.atom_num, config.dueling)
    network.to(device)
    checkpoint = config.save_interval * 150
    show = False
    if config.atom_num > 1:
        vrange = torch.linspace(config.min_value, config.max_value, config.atom_num).to(device)

    while os.path.exists('./models/'+str(checkpoint)+'checkpoint.pth'):
        network.load_state_dict(torch.load('./models/'+str(checkpoint)+'checkpoint.pth'))
        torch.save(network.state_dict(), './models/rainbow{}.pth'.format(checkpoint))

        sum_reward = 0
        round = 5
        for _ in range(round):
            obs = env.reset()
            done = False
            while not done:
                if show:
                    env.render()
                    time.sleep(0.05)
                obs = transform(obs).unsqueeze(0)

                q = network(obs)
                if config.atom_num > 1:
                    q = (q.exp() * vrange).sum(2)

                if random.random() < 0.01:
                    action = env.action_space.sample()
                else:
                    action = q.argmax(1).item()

                obs, reward, done, _ = env.step(action)
                sum_reward += reward

            env.close()

        print(' checkpoint: {}' .format(checkpoint))
        print(' avg reward: {}' .format(sum_reward/round))
        checkpoint += config.save_interval





if __name__ == '__main__':
    
    # test(args.env_name)



    env = gym.make('MsPacman-v0')
    obs = env.reset()
    plt.imshow(obs)
    # img.show()


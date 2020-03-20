import argparse
import config
from dqn import CNN
import gym
import torch
import os
import time
from torchvision import transforms

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
# parser.add_argument('--algorithm', type=str, help='Algorithm')
parser.add_argument('--env_name', type=str, default='MsPacman-v0', help='environment name')
parser.add_argument('--checkpoint', type=str, help='Algorithm')
args = parser.parse_args()

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((44, 44)),
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
])

def test(env_name, checkpoint):
    env = gym.make(env_name)
    network = CNN(config.input_shape, env.action_space.n, config.atom_num, config.dueling)
    if checkpoint:
        network.load_state_dict(torch.load(checkpoint+'.checkpoint')[0])
    else:
        last_checkpoint = config.save_interval
        while os.path.exists('./'+str(last_checkpoint)+'.checkpoint'):
            last_checkpoint += config.save_interval

        last_checkpoint -= config.save_interval

        network.load_state_dict(torch.load(str(last_checkpoint)+'.checkpoint')[0])

    test = 100
    sum_reward = 0
    for i in range(test):
        obs = env.reset()
        done = False
        while not done:
            if i  == test // 2 - 1:
                env.render()
                time.sleep(0.05)
            obs = trans(obs).unsqueeze(0) / 255

            val = network(obs)
            action = val.argmax(1).item()

            obs, reward, done, _ = env.step(action)
            sum_reward += reward

        env.close()

    print(' avg reward: {}' .format(sum_reward/test))





if __name__ == '__main__':
    
    test(args.env_name, args.checkpoint)




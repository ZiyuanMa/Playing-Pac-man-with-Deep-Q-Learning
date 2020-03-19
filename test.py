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
parser.add_argument('--env_name', type=str, required=True, help='environment name')
parser.add_argument('--checkpoint', type=str, help='Algorithm')
args = parser.parse_args()

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((44, 44)),
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
])

def test(args):
    env = gym.make(args.env_name)
    network = CNN(config.input_shape, env.action_space.n, 1, True)
    if args.checkpoint:
        network.load_state_dict(torch.load(args.checkpoint+'.checkpoint'))
    else:
        last_checkpoint = config.save_interval
        while os.path.exists('./'+str(last_checkpoint)+'.checkpoint'):
            last_checkpoint += config.save_interval

        last_checkpoint -= config.save_interval

        network.load_state_dict(torch.load(str(last_checkpoint)+'.checkpoint'))


    obs = env.reset()
    done = False
    t = 0
    while not done:
        env.render()
        obs = trans(obs).unsqueeze(0)

        network(obs)

        obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
        
        time.sleep(0.5)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()





if __name__ == '__main__':
    print(args.checkpoint)




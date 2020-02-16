import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import gym
from tqdm import tqdm
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
])


env = gym.make('MsPacman-v0')

env.reset()
for t in range(1):
    # env.render()
    
    obs, reward, done, info = env.step(env.action_space.sample()) # take a random action
    img = np.array(obs)
    img = img[0:195,:,:]
    img = trans(img)

    
    time.sleep(0.5)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break
env.close()


class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''
    def __init__(self):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            z_dim: A integer indicating the latent dimension.
        '''

        super().__init__()
        # Batch x 3 X 160 x 160
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, 10, 5),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU()
                    )

        # Batch x 32 X 31 x 31
        self.conv2 = nn.Sequential(
                        nn.Conv2d(32, 64, 6, 3, 1),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU()
                    )
        # Batch x 64 X 10 x 10
        self.conv3 = nn.Sequential(
                        nn.Conv2d(64, 128, 4, 2),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU()
                    )

        # Batch x 128 X 4 x 4
        self.conv4 = nn.Sequential(
                        nn.Conv2d(128, 256, 3),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU()
                    )

        # Batch x 256 X 2 x 2
        self.mu = nn.Linear(1024, 1024)
        self.var = nn.Linear(1024, 1024)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Flatten(x)
        z_mu = self.mu(x)
        z_var = self.var(x)

        return z_mu, z_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        # Batch x 1024 x 1 x 1
        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(1024, 128, 4),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU()
                    )

        # Batch x 128 x 4 x 4
        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 5, 3),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU()
                    )
        
        # Batch x 64 x 14 x 14
        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(64, 32, 5, 2),
                        nn.LeakyReLU()
                    )

        # Batch x 32 X 31 x 31
        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(32, 3, 10, 5),
                        nn.Sigmoid()
                    )

        # Batch x 3 X 160 x 160

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        return x

    
class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self):
        super().__init__()

        self.enc = Encoder()
        self.dec = Decoder()

    def forward(self, x):
        # encode
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        z = z_mu + z_var * torch.randn_like(z_var)

        # decode
        predicted = self.dec(z)

        return predicted, z_mu, z_var

class VAEDataset(Dataset):
    def __init__(self, size=65536):
        self.img = []
        self.size = size
    def __getitem__(self, index):
        return np.copy(self.img[index]), np.copy(self.img[index])

    def __len__(self):
        return len(self.img)
    
    def push(self, img):
        if len(self.img) < self.size:
            self.img.append(img)



def collect_data(dataset):
    env = gym.make('MsPacman-v0')
    pbar = tqdm(total=dataset.size)

    while len(dataset) < dataset.size:

        done = False
        env.reset()
        while not done:
        
            obs, _, done, _ = env.step(env.action_space.sample()) # take a random action
            img = np.array(obs)
            img = img[0:195,:,:]
            img = trans(img)
            dataset.push(img)
            pbar.update()

    env.close()

    
def train_VAE(dataset):
    network = VAE()
    network = network.to(device)
    optimizer = torch.optim.AdamW(network.parameters())

    for i in range(5):
        data_loader = DataLoader(dataset, 1024, shuffle=True)
        total_loss = 0
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            out = network(X)

            loss = torch.sum((out-y)**2)/76800
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('loss: %.4f' % total_loss)
    
            
if __name__ == '__main__':
    dataset = Dataset()
    collect_data(dataset)
    train_VAE(dataset)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, 3, 1, 1),
            nn.BatchNorm2d(filters)
        )

    def forward(self, x):
        return F.leaky_relu(x + (self.block(x)))


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(4, config.kernel_num, 3)
        self.conv_blocks = nn.ModuleList([ConvBlock(config.kernel_num, config.kernel_num, 3) for _ in range(8)])
        # output 2*2*64

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_blocks(x)
        return x

class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2*2*64, 2*2*64)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2*2*64, config.action_space)


    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)

        return x

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_net = ConvNet()
        self.flatten = nn.Flatten()
        self.fc_net = FCNet()
    
    def forward(self, x):
        x = self.conv_net(x)
        x = self.flatten(x)
        x = self.fc_net(x)
        return x

if __name__ == '__main__':
    a = torch.Tensor(10).uniform_(0,2)
    print(a[0].item())
    Network()
'''Neural network model'''
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import config


class Network(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        # 84 x 84 input
        self.feature = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(True),
            nn.Flatten(),
        )

        self.advantage = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(True),
            nn.Linear(512, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def step(self, obs):
        latent = self.feature(obs)
        qvalue = self.advantage(latent)
        svalue = self.value(latent)
        qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
        action = torch.argmax(qvalue, 1).item()

        return action, qvalue

    @autocast()
    def forward(self, x):
        latent = self.feature(x)
        qvalue = self.advantage(latent)
        svalue = self.value(latent)
        qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)

        return qvalue

import torch.nn as nn
from torch.nn.functional import log_softmax
from torchvision import transforms
import config

# input pre-processing
transform = transforms.Compose([
    transforms.Lambda(lambda x: x[:195,:,:]),
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.numpy()),
])


class Network(nn.Module):
    def __init__(self, action_dim, dueling=config.dueling, atom_num=config.atom_num):
        super().__init__()

        self.atom_num = atom_num

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

        self.q = nn.Sequential(
            nn.Linear(3136, 256),
            nn.ReLU(True),
            nn.Linear(256, action_dim * self.atom_num)
        )

        if dueling:
            self.state = nn.Sequential(
                nn.Linear(3136, 256),
                nn.ReLU(True),
                nn.Linear(256, self.atom_num)
            )

        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)
        latent = self.feature(x)
        qvalue = self.q(latent)
        if self.atom_num == 1:
            if hasattr(self, 'state'):
                svalue = self.state(latent)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            return qvalue
        else:
            qvalue = qvalue.view(batch_size, -1, self.atom_num)
            if hasattr(self, 'state'):
                svalue = self.state(latent).unsqueeze(1)
                qvalue = svalue + qvalue - qvalue.mean(1, keepdim=True)
            logprobs = log_softmax(qvalue, -1)
            return logprobs
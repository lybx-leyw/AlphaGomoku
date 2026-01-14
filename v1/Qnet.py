#他就是net9
import torch.nn as nn
from collections import deque
import random

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 7, padding=3),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(4*15*15, 225)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

"""
棋盘设置:15*15
黑棋的位置(-1)，白棋的位置(1)，身份标识，全局视野。

卷积网络设计：
 / 说明：大卷积核有利于模型对棋子之间的连接性产生更深刻的认识
"""

import torch.nn as nn

class trick_net(nn.Module):
    def __init__(self):
        super(trick_net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2),
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

class value_net(nn.Module):
    def __init__(self):
        super(value_net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(6*15*15, 225),
            nn.LeakyReLU(),
            nn.Linear(225,3)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
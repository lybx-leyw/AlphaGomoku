# real49%
# 简单修改通道层后：real_plus_47% 记为net_6 权重1.1
# plus_2_47% 记为net_7 权重1.3
# plus_3_48% 记为net_8 权重1.2
import torch.nn as nn

class symplenet(nn.Module):
    def __init__(self):
        super(symplenet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, padding=2),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 4, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2,stride=1,padding=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(4*16*16, 225)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
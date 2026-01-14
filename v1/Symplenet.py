#%32
import torch.nn as nn

class symplenet(nn.Module):
    def __init__(self):
        super(symplenet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 225),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
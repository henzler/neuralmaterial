import torch
import torch.nn as nn


class RotationEncoder(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, padding=1), nn.LeakyReLU(), nn.InstanceNorm2d(1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, padding=1), nn.LeakyReLU(), nn.InstanceNorm2d(1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, padding=1), nn.LeakyReLU(), nn.InstanceNorm2d(1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(1, 1, 3, 1, padding=1), nn.LeakyReLU(), nn.InstanceNorm2d(1)
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.linear = nn.Linear(1 * 8 * 8, 2)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.001)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.avgpool(x)
        bs = x.shape[0]

        x = torch.tanh(self.linear(x.reshape(bs, -1)))

        return x
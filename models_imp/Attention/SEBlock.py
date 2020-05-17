import torch.nn as nn
import torch

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // r)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // r, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x)
        y = torch.flatten(y, start_dim=1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(x.size(0), x.size(1), 1, 1)
        return x * y

if __name__ == '__main__':
    model = SEBlock(64, 64)
    features = torch.randn((4, 64, 7, 7))
    se_features = model(features)
    print(se_features.size())
import torch
import torch.nn as nn

class MobileNet(torch.nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # define network
        # general part: conv + bn + relu
        def conv_bn(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)       
            )

        def deep_wise_conv(in_channels, out_channels, stride):
            return nn.Sequential(
                # deepwise conv : in_channels = out_channels, with padding=1
                # implemented by group param
                nn.Conv2d(in_channels, in_channels, 3, stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),

                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)

            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            deep_wise_conv(32, 64, 1),
            deep_wise_conv(64, 128, 2),
            deep_wise_conv(128, 128, 1),
            deep_wise_conv(128, 256, 2),
            deep_wise_conv(256, 256, 1),
            deep_wise_conv(256, 512, 2),

            deep_wise_conv(512, 512, 1),
            deep_wise_conv(512, 512, 1),
            deep_wise_conv(512, 512, 1),
            deep_wise_conv(512, 512, 1),
            deep_wise_conv(512, 512, 1),

            deep_wise_conv(512, 1024, 2),
            deep_wise_conv(1024, 1024, 1),

            nn.AvgPool2d(7)

        )

        self.fc = nn.Linear(1024, 1000)
    
    def forward(self, x):
        x = self.model(x)
        x = nn.Flatten(x, start_dim=1)
        x = self.fc(x)
        return x
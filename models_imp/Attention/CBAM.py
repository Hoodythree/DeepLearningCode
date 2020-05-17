import torch.nn as nn
import torch

class CBAM(nn.Module):
    def __init__(self, in_channels, kernel_size=7, r=16):
        super(CBAM, self).__init__()
        self.ch_attention = CBAM_channel_att(in_channels, r)
        self.sp_attention = CBAM_spatial_att(kernel_size)

    def forward(self, x):
        # stage1 : channel attention
        ch_att = self.ch_attention(x)

        # stage2: spatial attention
        sp_att = self.sp_attention(x * ch_att)
        return x * sp_att


class CBAM_channel_att(nn.Module):
    def __init__(self, channels, r):
        super(CBAM_channel_att, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        x1, x2 = self.avg_pool(x).view(b, c), self.max_pool(x).view(b, c)
        x = self.fc(x1).view(b, c, 1, 1) + self.fc(x2).view(b, c, 1, 1)
        x = self.sigmoid(x)
        return x


class CBAM_spatial_att(nn.Module):
    def __init__(self, kernel_size):
        super(CBAM_spatial_att, self).__init__()
        assert kernel_size in [3, 7], 'kernel size must be 3 or 7'
        padding = 1 if kernel_size == 3 else 3
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # pooling cross channel
        avg_res = torch.mean(x, dim=1, keepdim=True)
        max_res, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_res, max_res], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    model = CBAM(64)
    features = torch.randn((4, 64, 7, 7))
    features = model(features)
    print('CBAM attention size : ', features.size())



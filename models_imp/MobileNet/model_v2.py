import torch
import torch.nn as nn

def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch
    
class ConBNRelu(nn.Sequential):
    # It's normal conv when groups=1
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConBNRelu, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


class InvertedResdual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResdual, self).__init__()
        # output channels
        hidden_channel = in_channels * expand_ratio
        self.use_shortcut = stride==1 and in_channels==out_channels

        layers = []
        # if expand_ratio > 1, then drop 1x1 conv
        if expand_ratio != 1:
            layers.append(ConBNRelu(in_channels, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv : stride??
            ConBNRelu(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            nn.Conv2d(hidden_channel, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(v)
        else:
            return self.conv(v)




class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResdual
        # 通过超参数alpha扩展（收缩）channel
        input_channels = _make_divisible(32 * alpha, round_nearest)
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        features = []
        # in_ch = 3(RGB)
        features.append(ConBNRelu(3, input_channels, stride=2, groups=1))

        inverted_settings = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        for t, c, n, s in inverted_settings:
            # n bottlenecks within a block
            output_channels = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                # 每一个bottleneck的输出要乘以扩展因子，因此in_ch != out_ch
                features.append(block(input_channels, output_channels, stride, t))
                input_channels = output_channels
        features.append(ConBNRelu(input_channels, last_channel, kernel_size=1, stride=1))
        self.features = nn.Sequential(*features)

        # classifier
        self.avepool = nn.AdaptiveAvgPool2d((1, 1))
        self.classfier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avepool(x)
        x = torch.flatten(x, 1)
        x = self.classfier(x)
        return x










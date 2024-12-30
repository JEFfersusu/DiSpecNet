import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x

def activation_fn(features, name='prelu', inplace=True):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(features)
    else:
        raise NotImplementedError('Activation function not implemented')

class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, act_name='prelu'):
        super().__init__()
        padding = int((kSize - 1) / 2) * dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(nOut),
            activation_fn(features=nOut, name=act_name)
        )

    def forward(self, x):
        return self.cbr(x)

class BR(nn.Module):
    def __init__(self, nOut, act_name='prelu'):
        super().__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(nOut),
            activation_fn(nOut, name=act_name)
        )

    def forward(self, x):
        return self.br(x)

class DICE(nn.Module):
    def __init__(self, channel_in, channel_out, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True):
        super().__init__()
        self.conv_channel = nn.Conv2d(channel_in, channel_in, kernel_size, groups=channel_in)
        self.conv_width = nn.Conv2d(width, width, kernel_size, groups=width)
        self.conv_height = nn.Conv2d(height, height, kernel_size, groups=height)
        self.br_act = BR(3 * channel_in)
        self.weight_avg_layer = CBR(3 * channel_in, channel_in, kSize=1, groups=channel_in)
        self.proj_layer = CBR(channel_in, channel_out, kSize=3, groups=math.gcd(channel_in, channel_out))
        self.vol_shuffle = Shuffle(3)

    def forward(self, x):
        out_ch_wise = self.conv_channel(x)
        out_h_wise = self.conv_height(x)
        out_w_wise = self.conv_width(x)
        outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise), 1)
        outputs = self.br_act(outputs)
        if self.vol_shuffle:
            outputs = self.vol_shuffle(outputs)
        outputs = self.weight_avg_layer(outputs)
        return self.proj_layer(outputs)

class StridedDICE(nn.Module):
    def __init__(self, channel_in, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True):
        super().__init__()
        self.left_layer = nn.Sequential(
            CBR(channel_in, channel_in, 3, stride=2, groups=channel_in),
            CBR(channel_in, channel_in, 1, 1)
        )
        self.right_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            DICE(channel_in, channel_in, height, width, kernel_size, dilation, shuffle),
            CBR(channel_in, channel_in, 1, 1)
        )
        self.shuffle = Shuffle(groups=2)

    def forward(self, x):
        x_left = self.left_layer(x)
        x_right = self.right_layer(x)
        concat = torch.cat([x_left, x_right], 1)
        return self.shuffle(concat)

class ShuffleDICEBlock(nn.Module):
    def __init__(self, inplanes, outplanes, height, width, c_tag=0.5, groups=2):
        super().__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.layer_right = nn.Sequential(
            CBR(self.right_part_in, self.right_part_out, 1, 1),
            DICE(self.right_part_out, self.right_part_out, height, width)
        )
        self.shuffle = Shuffle(groups=2)

    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        right = self.layer_right(right)
        return self.shuffle(torch.cat((left, right), 1))

class DiSpecNet(nn.Module):
    def __init__(self, channels_in=3, num_classes=7, s=0.2):
        super().__init__()
        sc_ch_dict = {
            0.2: [16, 16, 32, 64, 128, 1024],
            0.5: [24, 24, 48, 96, 192, 1024]
        }
        out_channel_map = sc_ch_dict[s]
        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)
        self.level3 = nn.Sequential(
            StridedDICE(out_channel_map[1], 56, 56),
            ShuffleDICEBlock(2 * out_channel_map[1], out_channel_map[2], 56, 56)
        )
        self.level4 = nn.Sequential(
            StridedDICE(out_channel_map[2], 28, 28),
            ShuffleDICEBlock(2 * out_channel_map[2], out_channel_map[3], 28, 28)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Conv2d(out_channel_map[3], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.level1(x)
        x = self.level3(x)
        x = self.level4(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

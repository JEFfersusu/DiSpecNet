import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Shuffle(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
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
        NotImplementedError('Not implemented yet')
        exit()
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
        assert len(dilation) == 3
        padding_1 = int((kernel_size - 1) / 2) * dilation[0]
        padding_2 = int((kernel_size - 1) / 2) * dilation[1]
        padding_3 = int((kernel_size - 1) / 2) * dilation[2]
        self.conv_channel = nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, groups=channel_in,
                                      padding=padding_1, bias=False, dilation=dilation[0])
        self.conv_width = nn.Conv2d(width, width, kernel_size=kernel_size, stride=1, groups=width,
                                    padding=padding_2, bias=False, dilation=dilation[1])
        self.conv_height = nn.Conv2d(height, height, kernel_size=kernel_size, stride=1, groups=height,
                                     padding=padding_3, bias=False, dilation=dilation[2])
        # self.global_filter = GlobalFilter(channel_in,height, width)
        self.br_act = BR(3 * channel_in)
        self.weight_avg_layer = CBR(3 * channel_in, channel_in, kSize=1, stride=1, groups=channel_in)
        groups_proj = math.gcd(channel_in, channel_out)
        self.proj_layer = CBR(channel_in, channel_out, kSize=3, stride=1, groups=groups_proj)
        self.linear_comb_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(channel_in, channel_in // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in // 4, channel_out, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.vol_shuffle = Shuffle(3)
        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.shuffle = shuffle
        self.ksize = kernel_size
        self.dilation = dilation
    def forward(self, x):
        bsz, channels, height, width = x.size()
        out_ch_wise = self.conv_channel(x)
        # out_gl_wise = self.global_filter(x)
        x_h_wise = x.clone()
        if height != self.height:
            if height < self.height:
                x_h_wise = F.interpolate(x_h_wise, mode='bilinear', size=(self.height, width), align_corners=True)
            else:
                x_h_wise = F.adaptive_avg_pool2d(x_h_wise, output_size=(self.height, width))
        x_h_wise = x_h_wise.transpose(1, 2).contiguous()
        out_h_wise = self.conv_height(x_h_wise).transpose(1, 2).contiguous()
        h_wise_height = out_h_wise.size(2)
        if height != h_wise_height:
            if h_wise_height < height:
                out_h_wise = F.interpolate(out_h_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_h_wise = F.adaptive_avg_pool2d(out_h_wise, output_size=(height, width))
        x_w_wise = x.clone()
        if width != self.width:
            if width < self.width:
                x_w_wise = F.interpolate(x_w_wise, mode='bilinear', size=(height, self.width), align_corners=True)
            else:
                x_w_wise = F.adaptive_avg_pool2d(x_w_wise, output_size=(height, self.width))
        x_w_wise = x_w_wise.transpose(1, 3).contiguous()
        out_w_wise = self.conv_width(x_w_wise).transpose(1, 3).contiguous()
        w_wise_width = out_w_wise.size(3)
        if width != w_wise_width:
            if w_wise_width < width:
                out_w_wise = F.interpolate(out_w_wise, mode='bilinear', size=(height, width), align_corners=True)
            else:
                out_w_wise = F.adaptive_avg_pool2d(out_w_wise, output_size=(height, width))
        # outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise,magnitude_spectrum), 1)
        outputs = torch.cat((out_ch_wise, out_h_wise, out_w_wise), 1)
        outputs = self.br_act(outputs)
        # print(outputs.shape)
        if self.shuffle:
            outputs = self.vol_shuffle(outputs)
        outputs = self.weight_avg_layer(outputs)
        linear_wts = self.linear_comb_layer(outputs)
        proj_out = self.proj_layer(outputs)
        return proj_out * linear_wts
    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, vol_shuffle={shuffle}, ' \
            'width={width}, height={height}, dilation={dilation})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
class StridedDICE(nn.Module):
    def __init__(self, channel_in, height, width, kernel_size=3, dilation=[1, 1, 1], shuffle=True):
        super().__init__()
        assert len(dilation) == 3
        self.left_layer = nn.Sequential(CBR(channel_in, channel_in, 3, stride=2, groups=channel_in),
                                        CBR(channel_in, channel_in, 1, 1)
                                        )
        self.right_layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, padding=1, stride=2),
            DICE(channel_in, channel_in, height, width, kernel_size=kernel_size, dilation=dilation,
                 shuffle=shuffle),
            CBR(channel_in, channel_in, 1, 1)
        )
        self.shuffle = Shuffle(groups=2)
        self.width = width
        self.height = height
        self.channel_in = channel_in
        self.channel_out = 2 * channel_in
        self.ksize = kernel_size
    def forward(self, x):
        x_left = self.left_layer(x)
        x_right = self.right_layer(x)
        concat = torch.cat([x_left, x_right], 1)
        return self.shuffle(concat)
    def __repr__(self):
        s = '{name}(in_channels={channel_in}, out_channels={channel_out}, kernel_size={ksize}, ' \
            'width={width}, height={height})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
class ShuffleDICEBlock(nn.Module):
    def __init__(self, inplanes, outplanes, height, width, c_tag=0.5, groups=2):
        super(ShuffleDICEBlock, self).__init__()
        self.left_part = round(c_tag * inplanes)
        self.right_part_in = inplanes - self.left_part
        self.right_part_out = outplanes - self.left_part
        self.layer_right = nn.Sequential(
            CBR(self.right_part_in, self.right_part_out, 1, 1),
            DICE(channel_in=self.right_part_out, channel_out=self.right_part_out, height=height, width=width)
        )
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.groups = groups
        self.shuffle = Shuffle(groups=2)
    def forward(self, x):
        left = x[:, :self.left_part, :, :]
        right = x[:, self.left_part:, :, :]
        right = self.layer_right(right)
        return self.shuffle(torch.cat((left, right), 1))
    def __repr__(self):
        s = '{name}(in_channels={inplanes}, out_channels={outplanes})'
        return s.format(name=self.__class__.__name__, **self.__dict__)
sc_ch_dict = {
    0.2: [16, 16, 32, 64, 128, 1024],
    0.5: [24, 24, 48, 96, 192, 1024],
    0.75: [24, 24, 86, 172, 344, 1024],
    1.0: [24, 24, 116, 232, 464, 1024],
    1.25: [24, 24, 144, 288, 576, 1024],
    1.5: [24, 24, 176, 352, 704, 1024],
    1.75: [24, 24, 210, 420, 840, 1024],
    2.0: [24, 24, 244, 488, 976, 1024],
    2.4: [24, 24, 278, 556, 1112, 1280],
}
class GlobalFilter(nn.Module):
    def __init__(self, C, H, W):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(C, H, W, dtype=torch.complex64) * 0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)
        magnitude_spectrum = 20 * torch.log(torch.abs(fshift) + 1e-6)
        # weight = self.complex_weight.expand(B, -1, -1, -1)
        # x_filtered_fft = fshift * weight
        x_filtered = torch.fft.ifft2(torch.fft.ifftshift(fshift))
        img_back = torch.abs(x_filtered)
        return img_back, magnitude_spectrum  
class CNNModel(nn.Module):
    def __init__(self, channels_in=3,num_classes=12, s=0.2, reps_at_each_level=[0, 3, 7, 3],each_cl_loss=False,glo=False):
        self.each_cl_loss = each_cl_loss
        self.glo = glo
        self.captured_tensors = {}
        width = height = 224
        super(CNNModel, self).__init__()
        out_channel_map = sc_ch_dict[s]
        assert width % 32 == 0, 'Input image width should be divisible by 32'
        assert height % 32 == 0, 'Input image height should be divisible by 32'
        width = int(width / 2)
        height = int(height / 2)
        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)
        self.fc1 = nn.Linear(9, num_classes)
        self.relu = nn.ReLU(inplace=True)
        if each_cl_loss:
            if glo:
                self.glo_1 = GlobalFilter(C=3, H=224, W=224)
                # self.glo_2 = GlobalFilter(C=16, H=56, W=56)
                self.glo_3 = GlobalFilter(C=32, H=56, W=56)
                self.glo_4 = GlobalFilter(C=64,H=28, W=28)
                self.glo_5 = GlobalFilter(C=128, H=14, W=14)
                self.o_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                if s == 0.2:
                    self.conv1x1_1 = nn.Conv2d(8, 3, kernel_size=1) 
                    # self.conv1x1_2 = nn.Conv2d(8, 8, kernel_size=1) 
                    # self.conv1x1_2 = nn.Conv2d(16, 16, kernel_size=1)
                else:
                    self.conv1x1_1 = nn.Conv2d(12, 3, kernel_size=1) 
            # self.fc1 = nn.Linear(6, num_classes)

        width = int(width / 2)
        height = int(height / 2)
        self.level2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        if each_cl_loss:
            self.fc2 = nn.Linear(int(out_channel_map[1]*3), num_classes)

        width = int(width / 2)
        height = int(height / 2)
        level3 = nn.ModuleList()
        level3.append(StridedDICE(channel_in=out_channel_map[1], height=height, width=width))
        for i in range(reps_at_each_level[1]):
            if i == 0:
                level3.append(ShuffleDICEBlock(2 * out_channel_map[1], out_channel_map[2], width=width, height=height))
            else:
                level3.append(ShuffleDICEBlock(out_channel_map[2], out_channel_map[2], width=width, height=height))
        self.level3 = nn.Sequential(*level3)
        if each_cl_loss:
            self.fc3 = nn.Linear(int(out_channel_map[2]*3), num_classes)

        level4 = nn.ModuleList()
        width = int(width / 2)
        height = int(height / 2)
        level4.append(StridedDICE(channel_in=out_channel_map[2], width=width, height=height))
        for i in range(reps_at_each_level[2]):
            if i == 0:
                level4.append(ShuffleDICEBlock(2 * out_channel_map[2], out_channel_map[3], width=width, height=height))
            else:
                level4.append(ShuffleDICEBlock(out_channel_map[3], out_channel_map[3], width=width, height=height))
        self.level4 = nn.Sequential(*level4)
        if each_cl_loss:
            self.fc4 = nn.Linear(int(out_channel_map[3]), num_classes)

        level5 = nn.ModuleList()
        width = int(width / 2)
        height = int(height / 2)
        level5.append(StridedDICE(channel_in=out_channel_map[3], width=width, height=height))
        for i in range(reps_at_each_level[3]):
            if i == 0:
                level5.append(ShuffleDICEBlock(2 * out_channel_map[3], out_channel_map[4], width=width, height=height))
            else:
                level5.append(ShuffleDICEBlock(out_channel_map[4], out_channel_map[4], width=width, height=height))
        self.level5 = nn.Sequential(*level5)
        if each_cl_loss:
            self.fc5 = nn.Linear(int(out_channel_map[4]), num_classes)

        if s > 1:
            self.drop_layer = nn.Dropout(p=0.2)
        else:
            self.drop_layer = nn.Dropout(p=0.1)

        groups = 4
        self.classifier = nn.Sequential(
            nn.Conv2d(out_channel_map[4], out_channel_map[5], kernel_size=1, groups=groups, bias=False),
            self.drop_layer,
            nn.Conv2d(out_channel_map[5], num_classes, 1, padding=0, bias=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.out_channel_map = out_channel_map
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        m = x
        x = self.level1(x)  # 112
        self.captured_tensors['x_level1'] = x.clone()
        if self.each_cl_loss:
            if self.glo:
                B, C, H, W = x.shape
                C_half = C // 2  
                x_half_1 = x[:, :C_half, :, :]  
                x_half_2 = x[:, C_half:, :, :]  
                filtered_image, magnitude_spectrum = self.glo_1(m)
                a_pooled1 = self.o_pool(magnitude_spectrum)
                a_pooled2 = self.o_pool(filtered_image)
                x_half_2_adjusted = self.relu(self.conv1x1_1(x_half_2))
                result1 = torch.cat((a_pooled1,x_half_2_adjusted), dim=1)
                result1 = torch.cat((result1, a_pooled2), dim=1)
                ex1 = F.avg_pool2d(result1, 112)
                ex1 = ex1.view(ex1.size(0), -1)
                ex1 = self.fc1(ex1)
                self.captured_tensors['filtered_image_level1'] = filtered_image.clone()
                self.captured_tensors['magnitude_spectrum_level1'] = magnitude_spectrum.clone()
            else:
                ex1 = x
                ex1 = F.avg_pool2d(ex1, 112)
                ex1 = ex1.view(ex1.size(0), -1)
                ex1 = self.fc1(ex1)

        # x = self.level2(x)  # 56
        
        x = self.level3(x)  # 28
        self.captured_tensors['x_level3'] = x.clone()
        if self.each_cl_loss:
            if self.glo:
                filtered_image, magnitude_spectrum = self.glo_3(x)
                self.captured_tensors['filtered_image_level3'] = filtered_image.clone()
                self.captured_tensors['magnitude_spectrum_level3'] = magnitude_spectrum.clone()
                result1 = torch.cat((magnitude_spectrum, x), dim=1)
                result1 = torch.cat((result1, filtered_image), dim=1)
                ex3 = F.avg_pool2d(result1, 56)
                ex3 = ex3.view(ex3.size(0), -1)
                ex3 = self.fc3(ex3)
            else:
                ex1 = x
                ex1 = F.avg_pool2d(ex1, 28)
                ex1 = ex1.view(ex1.size(0), -1)
                ex1 = self.fc3(ex1)
        x = self.level2(x)  # 56
        x = self.level4(x)  # 14
        self.captured_tensors['x_level4'] = x.clone()
        if self.each_cl_loss:
            if self.glo:
                # filtered_image, magnitude_spectrum = self.glo_4(x)
                # result1 = torch.cat((magnitude_spectrum, x), dim=1)
                # result1 = torch.cat((result1, filtered_image), dim=1)
                ex4 = F.avg_pool2d(x, 14)
                ex4 = ex4.view(ex4.size(0), -1)
                ex4 = self.fc4(ex4)
            else:
                ex1 = x
                ex1 = F.avg_pool2d(ex1, 14)
                ex1 = ex1.view(ex1.size(0), -1)
                ex1 = self.fc4(ex1)

        x = self.level5(x)  # 7
        self.captured_tensors['x_level5'] = x.clone()
        if self.each_cl_loss:
            if self.glo:
                # filtered_image, magnitude_spectrum = self.glo_5(x)
                # result1 = torch.cat((magnitude_spectrum, x), dim=1)
                # result1 = torch.cat((result1, filtered_image), dim=1)
                ex5 = F.avg_pool2d(x, 7)
                ex5 = ex5.view(ex5.size(0), -1)
                ex5 = self.fc5(ex5)
            else:
                ex1 = x
                ex1 = F.avg_pool2d(ex1, 112)
                ex1 = ex1.view(ex1.size(0), -1)
                ex1 = self.fc5(ex1)

        x = self.global_pool(x)
        x = self.classifier(x)
        
        x = x.view(x.size(0), -1)
        if self.each_cl_loss:
            return x,ex1,ex3,ex4,ex5
        return x

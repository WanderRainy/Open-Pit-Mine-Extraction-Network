"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F

from networks.convnext import ConvNeXt
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class MHNet(nn.Module):
    def __init__(self, num_classes):
        super(MHNet, self).__init__()

        self.num_classes=num_classes

        filters = [128, 256, 512, 1024]#[96, 192, 384, 768]
        self.backboon = ConvNeXt()
        self.backboon.load_state_dict(torch.load('./preweights/convnext_base_22k_224.pth')['model'], strict=False)
        
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

        self.height_deconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.height_relu1 = nonlinearity
        self.height_conv2 = nn.Conv2d(32, 32, 3)
        self.height_relu2 = nonlinearity
        self.height_conv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        # Encoder
        x4 = self.backboon(x)
        e4 = x4[3]
        e3 = x4[2]
        e2 = x4[1]
        e1 = x4[0]

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        height_out = self.height_deconv1(d1)
        height_out = self.height_relu1(height_out)
        height_out = self.height_conv2(height_out)
        height_out = self.height_relu2(height_out)
        height_out = self.height_conv3(height_out)

        return {'label_pred':torch.sigmoid(out),
                'height_pred':torch.sigmoid(height_out)}
        
import torch.nn as nn
import torch.nn.functional as F
import torch
from config import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # encoder
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3, 5), stride=(1, 2), padding=(1, 0))
        self.denseBlock1 = DenseBlock32()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.denseBlock2 = DenseBlock32()
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.denseBlock3 = DenseBlock32()
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.denseBlock4 = DenseBlock64()
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0))
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 0))

        # BLSTM
        self.BLSTM = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, bidirectional=True)

        # decoder
        self.deconv7 = nn.ConvTranspose2d(in_channels=1024, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                                          padding=(1, 0))
        self.deconv6 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.deconv5 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.dedenseBlock4 = DeDenseBlock64()
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.dedenseBlock3 = DeDenseBlock32()
        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.dedenseBlock2 = DeDenseBlock32()
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.dedenseBlock1 = DeDenseBlock32()
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(3, 5), stride=(1, 2),
                                          padding=(1, 0))

        # instance normalization
        self.IN32 = nn.InstanceNorm2d(num_features=32)
        self.IN64 = nn.InstanceNorm2d(num_features=64)
        self.IN128 = nn.InstanceNorm2d(num_features=128)
        self.IN256 = nn.InstanceNorm2d(num_features=256)
        self.IN512 = nn.InstanceNorm2d(num_features=512)

    def forward(self, x):
        # encoder
        en_x1 = self.conv1(x.permute(0, 3, 1, 2))
        en_x2 = self.denseBlock1(en_x1)
        en_x3 = F.elu(self.IN32(self.conv2(en_x2)))
        en_x4 = self.denseBlock2(en_x3)
        en_x5 = F.elu(self.IN32(self.conv3(en_x4)))
        en_x6 = self.denseBlock3(en_x5)
        en_x7 = F.elu(self.IN64(self.conv4(en_x6)))
        en_x8 = self.denseBlock4(en_x7)
        en_x9 = F.elu(self.IN128(self.conv5(en_x8)))
        en_x10 = F.elu(self.IN256(self.conv6(en_x9)))
        en_x11 = F.elu(self.IN512(self.conv7(en_x10)))

        # BLSTM
        self.BLSTM.flatten_parameters()
        x12, _ = self.BLSTM(en_x11.reshape(en_x11.size()[0], en_x11.size()[1], en_x11.size()[2]).permute(2, 0, 1))
        x12 = torch.mean(torch.stack(torch.split(x12.permute(1, 0, 2), split_size_or_sections=512, dim=2)), 0)
        x12 = x12.permute(0, 2, 1)
        x12 = x12.unsqueeze(dim=-1)
        # decoder
        de_x11 = F.elu(self.IN256(self.deconv7(torch.cat([x12, en_x11], dim=1))))
        de_x10 = F.elu(self.IN128(self.deconv6(torch.cat([de_x11, en_x10], dim=1))))
        de_x9 = F.elu(self.IN64(self.deconv5(torch.cat([de_x10, en_x9], dim=1))))
        de_x8 = self.dedenseBlock4(torch.cat([de_x9, en_x8], dim=1))
        de_x7 = F.elu(self.IN32(self.deconv4(de_x8)))
        de_x6 = self.dedenseBlock3(torch.cat([de_x7, en_x6], dim=1))
        de_x5 = F.elu(self.IN32(self.deconv3(de_x6)))
        de_x4 = self.dedenseBlock2(torch.cat([de_x5, en_x4], dim=1))
        de_x3 = F.elu(self.IN32(self.deconv2(de_x4)))
        de_x2 = self.dedenseBlock1(torch.cat([de_x3, en_x2], dim=1))
        de_x1 = self.deconv1(de_x2)

        return de_x1.permute(0, 2, 3, 1)


class DenseBlock32(nn.Module):
    def __init__(self):
        super(DenseBlock32, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=160, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.IN = nn.InstanceNorm2d(num_features=32)

    def forward(self, x):
        x1 = F.elu(self.IN(self.conv1(x)))
        x2 = F.elu(self.IN(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = F.elu(self.IN(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = F.elu(self.IN(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = F.elu(self.IN(self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))))
        return x5


class DenseBlock64(nn.Module):
    def __init__(self):
        super(DenseBlock64, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64 * 2, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64 * 4, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=64 * 5, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.IN = nn.InstanceNorm2d(num_features=64)

    def forward(self, x):
        x1 = F.elu(self.IN(self.conv1(x)))
        x2 = F.elu(self.IN(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = F.elu(self.IN(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = F.elu(self.IN(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = F.elu(self.IN(self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))))
        return x5


class DeDenseBlock64(nn.Module):
    def __init__(self):
        super(DeDenseBlock64, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=64 * 4, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64 * 5, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=64 * 6, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.IN = nn.InstanceNorm2d(num_features=64)

    def forward(self, x):
        x1 = F.elu(self.IN(self.conv1(x)))
        x2 = F.elu(self.IN(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = F.elu(self.IN(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = F.elu(self.IN(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = F.elu(self.IN(self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))))
        return x5


class DeDenseBlock32(nn.Module):
    def __init__(self):
        super(DeDenseBlock32, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32 * 3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=32 * 4, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=32 * 5, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = nn.Conv2d(in_channels=32 * 6, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.IN = nn.InstanceNorm2d(num_features=32)

    def forward(self, x):
        x1 = F.elu(self.IN(self.conv1(x)))
        x2 = F.elu(self.IN(self.conv2(torch.cat([x, x1], dim=1))))
        x3 = F.elu(self.IN(self.conv3(torch.cat([x, x1, x2], dim=1))))
        x4 = F.elu(self.IN(self.conv4(torch.cat([x, x1, x2, x3], dim=1))))
        x5 = F.elu(self.IN(self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))))
        return x5



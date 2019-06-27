import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

# -----------------------------deeplabv3-----------------------------------
class DeepLab(nn.Module):
    def __init__(self, num_classes=4, backbone='xception', output_stride=16, 
                 sync_bn=True, freeze_bn=False):
        super().__init__()
        if backbone == 'drn':
                    output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p




class DilatedConvBlock(nn.Module):
    ''' no dilation applied if dilation equals to 1 '''
    def __init__(self, in_size, out_size, kernel_size=3, dropout_rate=0.1, activation=F.relu, dilation=1):
        super().__init__()
        # to keep same width output, assign padding equal to dilation
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_size)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2, dilation=1):
        super().__init__()
        self.block1 = DilatedConvBlock(in_size, out_size, dropout_rate=0)
        self.block2 = DilatedConvBlock(out_size, out_size, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.pool(x), x

class ConvUpBlock(nn.Module):
    def __init__(self, in_size, out_size, dropout_rate=0.2, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_size, in_size//2, 2, stride=2)
        self.block1 = DilatedConvBlock(in_size//2 + out_size, out_size, dropout_rate=0)
        self.block2 = DilatedConvBlock(out_size, out_size, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, bridge):
        x = self.up(x)
        # align concat size by adding pad
        diffY = x.shape[2] - bridge.shape[2]
        diffX = x.shape[3] - bridge.shape[3]
        bridge = F.pad(bridge, (0, diffX, 0, diffY), mode='reflect')
        x = torch.cat([x, bridge], 1)
        # CAB: conv -> activation -> batch normal
        x = self.block1(x)
        x = self.block2(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # down conv
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv tunnel
        self.ce = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Dilated UNet
class DUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # down conv
        self.c1 = ConvBlock(3, 16, dilation=2)
        self.c2 = ConvBlock(16, 32, dilation=2)
        self.c3 = ConvBlock(32, 64, dilation=2)
        self.c4 = ConvBlock(64, 128, dilation=2)
        # bottom dilated conv tunnel
        self.d1 = DilatedConvBlock(128, 256)
        self.d2 = DilatedConvBlock(256, 256, dilation=2)
        self.d3 = DilatedConvBlock(256, 256, dilation=4)
        self.d4 = DilatedConvBlock(256, 256, dilation=8)
        # up conv
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # final conv tunnel
        self.ce = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Contour aware UNet
class CaUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # segmentation up conv branch
        self.u5s = ConvUpBlock(256, 128)
        self.u6s = ConvUpBlock(128, 64)
        self.u7s = ConvUpBlock(64, 32)
        self.u8s = ConvUpBlock(32, 16)
        self.ces = nn.Conv2d(16, 1, 1)
        # contour up conv branch
        self.u5c = ConvUpBlock(256, 128)
        self.u6c = ConvUpBlock(128, 64)
        self.u7c = ConvUpBlock(64, 32)
        self.u8c = ConvUpBlock(32, 16)
        self.cec = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        return xs, xc

# Contour aware Marker Unet
class CamUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        # segmentation up conv branch
        self.u5s = ConvUpBlock(256, 128)
        self.u6s = ConvUpBlock(128, 64)
        self.u7s = ConvUpBlock(64, 32)
        self.u8s = ConvUpBlock(32, 16)
        self.ces = nn.Conv2d(16, 1, 1)
        # contour up conv branch
        self.u5c = ConvUpBlock(256, 128)
        self.u6c = ConvUpBlock(128, 64)
        self.u7c = ConvUpBlock(64, 32)
        self.u8c = ConvUpBlock(32, 16)
        self.cec = nn.Conv2d(16, 1, 1)
        # marker up conv branch
        self.u5m = ConvUpBlock(256, 128)
        self.u6m = ConvUpBlock(128, 64)
        self.u7m = ConvUpBlock(64, 32)
        self.u8m = ConvUpBlock(32, 16)
        self.cem = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        xm = F.sigmoid(xm)
        return xs, xc, xm

# Contour aware marker Dilated Unet
class CamDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16, dilation=2)
        self.c2 = ConvBlock(16, 32, dilation=2)
        self.c3 = ConvBlock(32, 64, dilation=2)
        self.c4 = ConvBlock(64, 128, dilation=2)
        # bottom dilated conv tunnel
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256, dilation=2)
        # segmentation up conv branch
        self.u5s = ConvUpBlock(256, 128)
        self.u6s = ConvUpBlock(128, 64)
        self.u7s = ConvUpBlock(64, 32)
        self.u8s = ConvUpBlock(32, 16)
        self.ces = nn.Conv2d(16, 1, 1)
        # contour up conv branch
        self.u5c = ConvUpBlock(256, 128)
        self.u6c = ConvUpBlock(128, 64)
        self.u7c = ConvUpBlock(64, 32)
        self.u8c = ConvUpBlock(32, 16)
        self.cec = nn.Conv2d(16, 1, 1)
        # marker up conv branch
        self.u5m = ConvUpBlock(256, 128)
        self.u6m = ConvUpBlock(128, 64)
        self.u7m = ConvUpBlock(64, 32)
        self.u8m = ConvUpBlock(32, 16)
        self.cem = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        xm = F.sigmoid(xm)
        return xs, xc, xm


# Shared Contour aware Marker Unet
class SCamUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16)
        self.c2 = ConvBlock(16, 32)
        self.c3 = ConvBlock(32, 64)
        self.c4 = ConvBlock(64, 128)
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256)
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        self.ce = nn.Conv2d(16, 3, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = torch.split(x, 1, dim=1) # split 3 channels
        s = F.sigmoid(x[0])
        c = F.sigmoid(x[1])
        m = F.sigmoid(x[2])
        return s, c, m


# Shared Contour aware marker Dilated Unet
class SCamDUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16, dilation=2)
        self.c2 = ConvBlock(16, 32, dilation=2)
        self.c3 = ConvBlock(32, 64, dilation=2)
        self.c4 = ConvBlock(64, 128, dilation=2)
        # bottom dilated conv tunnel
        # bottom conv tunnel
        self.cu = ConvBlock(128, 256, dilation=2)
        self.u5 = ConvUpBlock(256, 128)
        self.u6 = ConvUpBlock(128, 64)
        self.u7 = ConvUpBlock(64, 32)
        self.u8 = ConvUpBlock(32, 16)
        # self.ce_s = nn.Conv2d(16, 2, 1)
        # self.ce_c = nn.Conv2d(16, 1, 1)
        # self.ce_m = nn.Conv2d(16, 1, 1)
        self.ce = nn.Conv2d(16, 4, 1)

    def forward(self, x):
        x, c1 = self.c1(x)
        x, c2 = self.c2(x)
        x, c3 = self.c3(x)
        x, c4 = self.c4(x)
        _, x = self.cu(x) # no maxpool for U bottom
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        # s = self.ce_s(x)
        # c = self.ce_c(x)
        # m = self.ce_m(x)
        x = self.ce(x)
        # x = torch.split(x, 1, dim=1) # split 3 channels
        # s = F.sigmoid(s)
        # c = F.sigmoid(c)
        # m = F.sigmoid(m)
        return x


# Transfer Learning VGG16_BatchNorm as Encoder part of UNet
class Vgg_UNet(nn.Module):
    def __init__(self, layers=16, fixed_feature=True):
        super().__init__()
        # load weight of pre-trained resnet
        self.vggnet = models.vgg16_bn(pretrained=True)
        # remove unused classifier submodule
        del self.vggnet.classifier
        self.vggnet.classifier = None
        # fine-tune or extract feature
        if fixed_feature:
            for param in self.vggnet.parameters():
                param.requires_grad = False
        # up conv
        self.u5 = ConvUpBlock(512, 512)
        self.u6 = ConvUpBlock(512, 256)
        self.u7 = ConvUpBlock(256, 128)
        self.u8 = ConvUpBlock(128, 64)
        # final conv tunnel
        self.ce = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        c = []
        for f in self.vggnet.features:
            if isinstance(f, nn.MaxPool2d):
                c.append(x)
            x = f(x)
        assert len(c) == 5
        x = self.u5(c[4], c[3])
        x = self.u6(x, c[2])
        x = self.u7(x, c[1])
        x = self.u8(x, c[0])
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Transfer Learning ResNet as Encoder part of UNet
class Res_UNet(nn.Module):
    def __init__(self, layers=34, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # up conv
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        self.ce = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        x = self.ce(x)
        x = F.sigmoid(x)
        return x

# Transfer Learning ResNet as Encoder part of Contour aware Marker Unet
class Res_CamUNet(nn.Module):
    def __init__(self, layers=34, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # segmentation up conv branch
        self.u5s = ConvUpBlock(l[4], l[3])
        self.u6s = ConvUpBlock(l[3], l[2])
        self.u7s = ConvUpBlock(l[2], l[1])
        self.u8s = ConvUpBlock(l[1], l[0])
        self.ces = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        # contour up conv branch
        self.u5c = ConvUpBlock(l[4], l[3])
        self.u6c = ConvUpBlock(l[3], l[2])
        self.u7c = ConvUpBlock(l[2], l[1])
        self.u8c = ConvUpBlock(l[1], l[0])
        self.cec = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        # marker up conv branch
        self.u5m = ConvUpBlock(l[4], l[3])
        self.u6m = ConvUpBlock(l[3], l[2])
        self.u7m = ConvUpBlock(l[2], l[1])
        self.u8m = ConvUpBlock(l[1], l[0])
        self.cem = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # segmentation up conv branch
        xs = self.u5s(x, c4)
        xs = self.u6s(xs, c3)
        xs = self.u7s(xs, c2)
        xs = self.u8s(xs, c1)
        xs = self.ces(xs)
        xs = F.sigmoid(xs)
        # contour up conv branch
        xc = self.u5c(x, c4)
        xc = self.u6c(xc, c3)
        xc = self.u7c(xc, c2)
        xc = self.u8c(xc, c1)
        xc = self.cec(xc)
        xc = F.sigmoid(xc)
        # marker up conv branch
        xm = self.u5m(x, c4)
        xm = self.u6m(xm, c3)
        xm = self.u7m(xm, c2)
        xm = self.u8m(xm, c1)
        xm = self.cem(xm)
        xm = F.sigmoid(xm)
        return xs, xc, xm

# Transfer Learning ResNet as Encoder part of Contour aware Marker Unet
class Res_SamUNet(nn.Module):
    def __init__(self, layers=101, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 101:
            builder = models.resnet101
            l = [64, 256, 512, 1024, 2048]
        else:
            builder = models.resnet34
            l = [64, 64, 128, 256, 512]
        # load weight of pre-trained resnet
        self.resnet = builder(pretrained=True)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # segmentation up conv branch
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        # self.ces = nn.ConvTranspose2d(l[0], 2, 2, stride=2)
        # self.cec = nn.ConvTranspose2d(l[0], 1, 2, stride=2)
        # self.cem = nn.ConvTranspose2d(l[0], 2, 2, stride=2)

        self.ce = nn.ConvTranspose2d(l[0], 5, 2, stride=2)
        self.ce_c = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = c1 = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        xs = self.ce(x)
        xc = self.ce_c(x)
        # xs = self.ces(x)
        # xs = F.sigmoid(xs)
        # xc = self.cec(x)
        # xc = F.sigmoid(xc)
        # xm = self.cem(x)
        # xm = F.sigmoid(xm)
        return torch.cat((xs,xc), 1)

# Transfer Learning DenseNet as Encoder part of UNet
class Dense_UNet(nn.Module):
    def __init__(self, layers=121, fixed_feature=True):
        super().__init__()
        # define pre-train model parameters
        if layers == 201:
            builder = models.densenet201
            l = [64, 256, 512, 1792, 1920]
        else:
            builder = models.densenet121
            l = [64, 256, 512, 1024, 1024]
        # load weight of pre-trained resnet
        self.densenet = builder(pretrained=True)
        if fixed_feature:
            for param in self.densenet.parameters():
                param.requires_grad = False
        # remove unused classifier submodule
        del self.densenet.classifier
        self.densenet.classifier = None
        # up conv
        self.u5 = ConvUpBlock(l[4], l[3])
        self.u6 = ConvUpBlock(l[3], l[2])
        self.u7 = ConvUpBlock(l[2], l[1])
        self.u8 = ConvUpBlock(l[1], l[0])
        # final conv tunnel
        self.ce = nn.ConvTranspose2d(l[0], 1, 2, stride=2)

    def forward(self, x):
        # refer https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
        c = []
        for f in self.densenet.features:
            if f.__class__.__name__ in ['MaxPool2d', '_Transition']:
                c.append(x)
            x = f(x)
        assert len(c) == 4
        x = self.u5(x, c[3])
        x = self.u6(x, c[2])
        x = self.u7(x, c[1])
        x = self.u8(x, c[0])
        x = self.ce(x)
        x = F.sigmoid(x)
        return x


# Deep Contour Aware Network (DCAN)
class dcanConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_ratio=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.Dropout2d(p=dropout_ratio),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class dcanDeConv(nn.Module):
    def __init__(self, in_ch, out_ch, upscale_factor, dropout_ratio=0.2):
        super().__init__()
        self.upscaling = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=upscale_factor, stride=upscale_factor)
        self.conv = dcanConv(out_ch, out_ch, dropout_ratio)

    def forward(self, x):
        x = self.upscaling(x)
        x = self.conv(x)
        return x

class DCAN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1 = dcanConv(n_channels, 64)
        self.conv2 = dcanConv(64, 128)
        self.conv3 = dcanConv(128, 256)
        self.conv4 = dcanConv(256, 512)
        self.conv5 = dcanConv(512, 512)
        self.conv6 = dcanConv(512, 1024)
        self.deconv3s = dcanDeConv(512, n_classes, 8) # 8 = 2^3 (3 maxpooling)
        self.deconv3c = dcanDeConv(512, n_classes, 8)
        self.deconv2s = dcanDeConv(512, n_classes, 16) # 16 = 2^4 (4 maxpooling)
        self.deconv2c = dcanDeConv(512, n_classes, 16)
        self.deconv1s = dcanDeConv(1024, n_classes, 32) # 32 = 2^5 (5 maxpooling)
        self.deconv1c = dcanDeConv(1024, n_classes, 32)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(self.maxpool(c1))
        c3 = self.conv3(self.maxpool(c2))
        c4 = self.conv4(self.maxpool(c3))
        # s for segment branch, c for contour branch
        u3s = self.deconv3s(c4)
        u3c = self.deconv3c(c4)
        c5 = self.conv5(self.maxpool(c4))
        u2s = self.deconv2s(c5)
        u2c = self.deconv2c(c5)
        c6 = self.conv6(self.maxpool(c5))
        u1s = self.deconv1s(c6)
        u1c = self.deconv1c(c6)
        outs = F.sigmoid(u1s + u2s + u3s)
        outc = F.sigmoid(u1c + u2c + u3c)
        return outs, outc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(model_name='unet'):
    # initialize model
    if model_name == 'unet':
        model = UNet()
    elif model_name == 'dcan':
        model = DCAN(3, 1)
    elif model_name == 'deeplab':
        model = DeepLab()
    elif model_name == 'caunet':
        model = CaUNet()
    elif model_name == 'camunet':
        model = CamUNet()
    elif model_name == 'camdunet':
        model = CamDUNet()
    elif model_name == 'scamunet':
        model = SCamUNet()
    elif model_name == 'scamdunet':
        model = SCamDUNet()
    elif model_name == 'vgg_unet':
        model = Vgg_UNet(16, fixed_feature=True)
    elif model_name == 'res_unet':
        model = Res_UNet(34, fixed_feature=True)
    elif model_name == 'dense_unet':
        model = Dense_UNet(121, fixed_feature=True)
    elif model_name == 'res_camunet':
        model = Res_CamUNet(34, fixed_feature=True)
    elif model_name == 'res_samunet':
        model = Res_SamUNet(34, fixed_feature=True)
    else:
        raise NotImplementedError()
    return model


if __name__ == '__main__':
    print('Network parameters -')
    for n in ['unet', 'camunet', 'scamunet', 'res_unet', 'res_camunet', 'res_samunet']:
        net = build_model(n)
        #print(net)
        print('\t model {}: {}'.format(n, count_parameters(net)))
        del net

    print("Forward pass sanity check - ")
    for n in ['camunet', 'res_camunet', 'res_samunet']:
        t = time.time()
        net = build_model(n)
        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        #print(x.shape, y.shape)
        del net
        print('\t model {0}: {1:.3f} seconds'.format(n, time.time() - t))

    # x = torch.randn(10, 3, 256, 256)
    # b = ConvBlock(3, 16)
    # p, y = b(x)
    # print(p.shape, y.shape)
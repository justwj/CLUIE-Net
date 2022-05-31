import torch
import torch.nn as nn
from torch.nn import functional as F
from models.unet_parts import *
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
class ResNet50BasicBlock(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(ResNet50BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + x)


class ResNet50DownBlock(nn.Module):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
        super(ResNet50DownBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm2d(outs[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(x_shortcut + out)


#dowm channel
class DClayer(nn.Module):
    def __init__(self, in_channel, outs,stride):
        super(DClayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=1, stride=stride[0], padding=0)
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=1, stride=stride[2], padding=0)
        self.bn3 = nn.BatchNorm2d(outs[2])
        self.conv4 = nn.Conv2d(outs[2], outs[3], kernel_size=3, stride=stride[3], padding=1)
        self.bn4 = nn.BatchNorm2d(outs[3])
        self.conv5 = nn.Conv2d(outs[3], outs[4], kernel_size=1, stride=stride[4], padding=0)
        self.bn5 = nn.BatchNorm2d(outs[4])
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = self.conv5(out)
        out = self.bn5(out)
        out = torch.tanh(out)
        return out


## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResNet50DownBlock(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
        )

        self.layer2 = nn.Sequential(
            ResNet50DownBlock(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
        )

        self.layer3 = nn.Sequential(
            ResNet50DownBlock(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1],padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1],padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0])
        )
        self.layer4 = nn.Sequential(
            ResNet50DownBlock(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            ResNet50BasicBlock(2048, outs=[512, 512, 2048], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
            ResNet50BasicBlock(2048, outs=[512, 512, 2048], kernerl_size=[1, 3, 1], stride=[1, 1, 1], padding=[0, 1, 0]),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.meanpool=nn.AvgPool2d((2,2),stride=(2,2))
        self.upsample=nn.Upsample(scale_factor=2,mode='nearest')

        self.fc1 = nn.Linear(3072, 1)
        self.fc = nn.Linear(1024, 1)
        self.CALayer1=CALayer(1024)
        self.CALayer2 = CALayer(2048)
        self.CALayer3 = CALayer(4096)
        self.DCLayer1=DClayer(1024,outs=[512,512,256,256,1],stride=[1,1,1,1,1])
        self.DCLayer2 = DClayer(2048, outs=[1024, 512, 256, 256, 1],stride=[1,1,1,1,1])
        self.DCLayer3 = DClayer(4096, outs=[2048, 1024, 512, 256, 1],stride=[1,1,1,1,1])
        #
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.8),
            nn.Linear(512, 1),
        )
    def forward(self, x1,x2):
        out1 = self.conv1(x1)
        out1 = self.maxpool(out1)
        out1_layer1 = self.layer1(out1)
        out1_layer2 = self.layer2(out1_layer1)

        out1_layer3= self.layer3(out1_layer2)

        out1_layer4 = self.layer4(out1_layer3)

        out2 = self.conv1(x2)
        out2 = self.maxpool(out2)
        out2_layer1 = self.layer1(out2)
        out2_layer2 = self.layer2(out2_layer1)

        out2_layer3 = self.layer3(out2_layer2)

        out2_layer4 = self.layer4(out2_layer3)



        low_disimg = torch.cat((out1_layer2, out2_layer2), 1)
        low_disimg = self.CALayer1(low_disimg)
        low_disimg = self.DCLayer1(low_disimg)


        mid_disimg = torch.cat((out1_layer3, out2_layer3), 1)
        mid_disimg = self.CALayer2(mid_disimg)
        mid_disimg = self.DCLayer2(mid_disimg)

        hig_disimg = torch.cat((out1_layer4, out2_layer4), 1)
        hig_disimg = self.CALayer3(hig_disimg)
        hig_disimg = self.DCLayer3(hig_disimg)

        disimg=(self.meanpool(low_disimg)+mid_disimg+self.upsample(hig_disimg))/3

        return low_disimg,mid_disimg,hig_disimg,disimg







# 2021/4/8
# author:wuli
# 本程序用于自定义Superiority Discriminative loss损失函数
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import os
import time
import cv2
from torchvision.utils import save_image
from models.unet_parts import *


def write_to_log(log_file_path, status):
    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')


def loss(transform, net_score, net_disimg, out, e):
    e = Variable(e).cuda()
    e = transform(e)
    # print(e.size())
    # e = e.unsqueeze(dim=0)
    one = (torch.ones(256, 256) * 0.5).cuda()
    start_time = time.time()
    score = net_score(out, e)
    end_time = time.time()
    # print(end_time-start_time)
    write_to_log('./1/score1.txt', str(score.item()))
    strat_time1 = time.time()
    disimg3 = torch.squeeze(net_disimg(out, e))
    end_time1 = time.time()
    # print(end_time1-strat_time1)
    if score.item() >= 0.5:
        loss = (torch.mean(one - disimg3).item()) * (score - 0.5)
    if score.item() <= 0.5:
        loss = (torch.mean(disimg3 - one).item()) * (score - 0.5)
    return loss






class SDLoss(nn.Module):
    """Calculate Superiority Discriminative loss"""

    def __init__(self):
        super(SDLoss, self).__init__()

    def forward(self, loss_net, out, loss1img, loss2img, loss3img, loss4img, loss5img, loss6img, loss7img, loss8img,
                loss9img, loss10img, loss11img):
        low_disimg1,mid_disimg1,hig_disimg1,disimg1 = loss_net(out, loss1img)
        low_disimg2,mid_disimg2,hig_disimg2,disimg2 = loss_net(out, loss2img)
        low_disimg3,mid_disimg3,hig_disimg3,disimg3 = loss_net(out, loss3img)
        low_disimg4,mid_disimg4,hig_disimg4,disimg4 = loss_net(out, loss4img)
        low_disimg5,mid_disimg5,hig_disimg5,disimg5 = loss_net(out, loss5img)
        low_disimg6,mid_disimg6,hig_disimg6,disimg6 = loss_net(out, loss6img)
        low_disimg7,mid_disimg7,hig_disimg7,disimg7 = loss_net(out, loss7img)
        low_disimg8,mid_disimg8,hig_disimg8,disimg8 = loss_net(out, loss8img)
        low_disimg9,mid_disimg9,hig_disimg9,disimg9 = loss_net(out, loss9img)
        low_disimg10,mid_disimg10,hig_disimg10,disimg10 = loss_net(out, loss10img)
        low_disimg11,mid_disimg11,hig_disimg11,disimg11 = loss_net(out, loss11img)
        #disimg12 = loss_net(out, loss12img)
        zero = torch.ones_like(disimg1)
        #zero = torch.zeros_like(disimg1)
        disimg1 = torch.where(disimg1 > 0, zero, zero - disimg1)
        disimg2 = torch.where(disimg2 > 0, zero, zero - disimg2)
        disimg3 = torch.where(disimg3 > 0, zero, zero - disimg3)
        disimg4 = torch.where(disimg4 > 0, zero, zero - disimg4)
        disimg5 = torch.where(disimg5 > 0, zero, zero - disimg5)
        disimg6 = torch.where(disimg6 > 0, zero, zero - disimg6)
        disimg7 = torch.where(disimg7 > 0, zero, zero - disimg7)
        disimg8 = torch.where(disimg8 > 0, zero, zero - disimg8)
        disimg9 = torch.where(disimg9 > 0, zero, zero - disimg9)
        disimg10 = torch.where(disimg10 > 0, zero, zero - disimg10)
        disimg11 = torch.where(disimg11 > 0, zero, zero - disimg11)

        #print(disimg1 + disimg2 + disimg3 + disimg4 + disimg5 + disimg6 + disimg7 + disimg8 + disimg9 + disimg10 + disimg11)
        loss = torch.mean(
            disimg1 + disimg2 + disimg3 + disimg4 + disimg5 + disimg6 + disimg7 + disimg8 + disimg9 + disimg10 + disimg11)
        #print(loss)
        return loss




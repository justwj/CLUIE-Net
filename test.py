import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import datetime
import sys
from models.network import ResNet50
from numpy import *
import random
from utils.config import opt
from utils.focal import focal_loss, WeightLoss
from datasets.dataset2 import TestDataset
import time
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr, compare_mse
import scipy.stats as stats
from tqdm import tqdm
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score


def write_to_log(log_file_path, status):
    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')


def test(path):
    score_correct = []
    disimg_correct1 = []
    input_dataset = TestDataset()
    input_dataloader = DataLoader(input_dataset, batch_size=4, shuffle=True, num_workers=opt.num_workers)
    net = ResNet50().cuda()
    net.load_state_dict(path)
    net.eval()
    for idx, (raw_image, tmp_image, score_gt, disimg_gt_train1, raw_names, tmp_names, disimg_gt_names) in tqdm(enumerate(input_dataloader)):
        raw_image = Variable(raw_image).cuda()
        tmp_image = Variable(tmp_image).cuda()
        score_gt = score_gt.float()
        score_gt = Variable(score_gt).cuda()
        disimg_gt_train1 = Variable(disimg_gt_train1).cuda()
        low_disimg,mid_disimg,hig_disimg,disimg = net(raw_image, tmp_image)
        disimg=torch.squeeze(disimg)
        for i in range(0, len(score_gt)):
            one = torch.ones_like(disimg_gt_train1[i])
            zero = torch.zeros_like(disimg_gt_train1[i])
            correct = torch.where((disimg[i] >= 0), one, (zero - one))
            if ((torch.sum(correct) > 0) and (score_gt[i] == 0)) or ((torch.sum(correct) < 0) and (score_gt[i] == 1)):
                score_correct.append(1)
            else:
                score_correct.append(0)
        for i in range(0, len(score_gt)):
            one = torch.ones_like(disimg_gt_train1[i])
            zero = torch.zeros_like(disimg_gt_train1[i])
            correct = torch.where((disimg[i].mul(disimg_gt_train1[i]) >= 0), one, zero)
            correct = torch.sum(correct) / (16 * 16)
            disimg_correct1.append(correct.item())
    score_acc = sum(score_correct) / len(score_correct)
    disimg_acc1 = sum(disimg_correct1) / len(disimg_correct1)
    print('The accuracy of scores is {}, and the accuracy of quality-superiority maps in MSE is {}'.format(score_acc,disimg_acc1))
    return score_acc,disimg_acc1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    continue_train = False
    path = torch.load(os.path.join('./checkpoint/QC/net_latest.pkl'))
    s_acc, d_acc = test(path)
    status = "disimg_mse_acc:{},score_acc:{}".format(d_acc, s_acc)
    print(status)




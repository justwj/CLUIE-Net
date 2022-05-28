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
from network import ResNet50
from numpy import *
import random
from utils.config import opt
from utils.focal import focal_loss, WeightLoss
from datasets.dataset2 import TrainDataset, EvalDataset
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


def Losscomputer(score, gt):
    criterion = WeightLoss.cuda()
    loss = criterion(score, gt)
    return loss


def train(train_name, continue_train):
    if not os.path.exists(opt.model_save_path):
        os.mkdir(opt.model_save_path)
    if not os.path.exists(os.path.join(opt.model_save_path, train_name)):
        os.mkdir(os.path.join(opt.model_save_path, train_name))
    log_file_path = os.path.join(opt.model_save_path, train_name, 'log_file.txt')
    input_dataset = TrainDataset()
    input_dataloader = DataLoader(input_dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers,
                                  pin_memory=True)
    net = ResNet50().cuda()
    net.train()
    lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=opt.weight_decay)
    if continue_train:
        net_load_path = './checkpoint/QC/net_idx15.pkl'
        net.load_state_dict(torch.load(net_load_path))
    D_loss = nn.MSELoss().cuda()
    for epoch in range(1, 101):
        start_time = time.time()
        for idx, (raw_image, tmp_image, score_gt, disimg_gt_train32,disimg_gt_train16,disimg_gt_train8,disimg_gt_train, raw_names, tmp_names, disimg_gt_names) in tqdm(enumerate(input_dataloader)):
            raw_image = Variable(raw_image).cuda()
            tmp_image = Variable(tmp_image).cuda()
            score_gt = score_gt.float()
            score_gt = Variable(score_gt).cuda()
            disimg_gt_train32 = Variable(disimg_gt_train32.float()).cuda()
            disimg_gt_train16 = Variable(disimg_gt_train16.float()).cuda()
            disimg_gt_train8 = Variable(disimg_gt_train8.float()).cuda()
            disimg_gt_train = Variable(disimg_gt_train.float()).cuda()
            low_disimg,mid_disimg,hig_disimg,disimg = net(raw_image, tmp_image)
            low_disimg=torch.squeeze(low_disimg)
            mid_disimg = torch.squeeze(mid_disimg)
            hig_disimg = torch.squeeze(hig_disimg)
            disimg = torch.squeeze(disimg)
            disimg_loss = D_loss(low_disimg, disimg_gt_train32)*5+D_loss(mid_disimg, disimg_gt_train16)*5+D_loss(hig_disimg, disimg_gt_train8)*5
            loss = disimg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (idx + 1) % opt.print_freq == 0:
                progress = "Epoch:{}-{},lr: {},low_loss:{},mid_loss:{},hig_loss:{},loss:{}".format(epoch, idx + 1,optimizer.param_groups[0]['lr'],D_loss(low_disimg, disimg_gt_train32)*5,D_loss(mid_disimg, disimg_gt_train16)*5,D_loss(hig_disimg, disimg_gt_train8)*5, loss)
                print(progress)
                write_to_log(log_file_path, progress)


        end_time = time.time()
        print('epoch {} use time:{}\n'.format(epoch, end_time - start_time))
        print('save Training model epoch' + str(epoch))
        torch.save(net.state_dict(), os.path.join(opt.model_save_path, train_name, 'net_idx' + str(epoch) + '.pkl'))
#         s_acc, d_acc = val(net)
#         status = "Epoch:{},disimg_mse_acc:{},score_acc:{}".format(epoch,d_acc,s_acc)
#         write_to_log('./checkpoint/evlacc.txt', status)
    print('Finished Training')
    torch.save(net.state_dict(), os.path.join(opt.model_save_path, train_name, 'net_latest' + '.pkl'))


def val(net):
    score_correct = []
    disimg_correct1 = []
    input_dataset = EvalDataset()
    input_dataloader = DataLoader(input_dataset, batch_size=8, shuffle=True, num_workers=opt.num_workers)
    net = ResNet50().cuda()
    net.load_state_dict(torch.load('./checkpoint/QC/net_latest.pkl'))
    net.eval()
    for idx, (raw_image, tmp_image, score_gt, disimg_gt_train1, raw_names, tmp_names, disimg_gt_names) in tqdm(enumerate(input_dataloader)):
        raw_image = Variable(raw_image).cuda()
        tmp_image = Variable(tmp_image).cuda()
        score_gt = score_gt.float()
        score_gt = Variable(score_gt).cuda()
        disimg_gt_train1 = Variable(disimg_gt_train1).cuda()
        low_disimg, mid_disimg, hig_disimg, disimg = net(raw_image, tmp_image)
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
    train_name = 'QC'
    continue_train = False
    train(train_name, continue_train)

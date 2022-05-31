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
import click
from models.network import ResNet50
from numpy import *
import random
from datasets.dataset import TrainDataset
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


@click.command()
@click.option('--train_path', default='./data/RQSD-UI/E_imgs')
@click.option('--txt_path', default='./data/RQSD-UI/train-validation_pairs/train-Cons.txt')
@click.option('--modelsave_path', default='./checkpoint')
@click.option('--batch_size', default=16)
@click.option('--num_workers', default=4)
@click.option('--weight_decay', default=0.01)
@click.option('--continue_train', default=False)
@click.option('--lambda1', default=5)
@click.option('--lambda2', default=5)
@click.option('--lambda3', default=5)
@click.option('--epoch', default=100)
@click.option('--print_freq', default=20)


def train(train_path,txt_path,modelsave_path,batch_size,num_workers,weight_decay,continue_train,lambda1,lambda2,lambda3,epoch,print_freq):
    if not os.path.exists(modelsave_path):
        os.mkdir(modelsave_path)
    log_file_path = os.path.join(modelsave_path, 'log_file.txt')
    input_dataset = TrainDataset(train_imgs_path=train_path, train_txt_path=txt_path)
    input_dataloader = DataLoader(input_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    net = ResNet50().cuda()
    net.train()
    lr = 1e-4
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    if continue_train:
        net_load_path = './checkpoint/net_latest.pkl'
        net.load_state_dict(torch.load(net_load_path))
    D_loss = nn.MSELoss().cuda()
    for i in range(1, epoch):
        start_time = time.time()
        for idx, (raw_image, tmp_image, score_gt, disimg_gt_train32,disimg_gt_train16,disimg_gt_train8, raw_names, tmp_names, disimg_gt_names) in tqdm(enumerate(input_dataloader)):
            raw_image = Variable(raw_image).cuda()
            tmp_image = Variable(tmp_image).cuda()
            disimg_gt_train32 = Variable(disimg_gt_train32.float()).cuda()
            disimg_gt_train16 = Variable(disimg_gt_train16.float()).cuda()
            disimg_gt_train8 = Variable(disimg_gt_train8.float()).cuda()
            low_disimg, mid_disimg, hig_disimg, disimg = net(raw_image, tmp_image)
            low_disimg = torch.squeeze(low_disimg)
            mid_disimg = torch.squeeze(mid_disimg)
            hig_disimg = torch.squeeze(hig_disimg)
            disimg_loss = D_loss(low_disimg, disimg_gt_train32)*lambda1+D_loss(mid_disimg, disimg_gt_train16)*lambda2+D_loss(hig_disimg, disimg_gt_train8)*lambda3
            loss = disimg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (idx + 1) % print_freq == 0:
                progress = "Epoch:{}-{},lr: {},low_loss:{},mid_loss:{},hig_loss:{},loss:{}".format(i, idx + 1,optimizer.param_groups[0]['lr'],D_loss(low_disimg, disimg_gt_train32)*lambda1,D_loss(mid_disimg, disimg_gt_train16)*lambda2,D_loss(hig_disimg, disimg_gt_train8)*lambda3, loss)
                print(progress)
                write_to_log(log_file_path, progress)


        end_time = time.time()
        print('epoch {} use time:{}\n'.format(i, end_time - start_time))
        print('save Training model epoch' + str(i))
        torch.save(net.state_dict(), os.path.join(model_save_path, 'net_idx' + str(i) + '.pkl'))
    print('Finished Training')
    torch.save(net.state_dict(), os.path.join(model_save_path, 'net_latest' + '.pkl'))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()

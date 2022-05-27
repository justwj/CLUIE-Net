from torch.utils.data.dataset import Dataset
import os
import torch
from torchvision import transforms
from PIL import Image
from utils.config import opt
from glob import glob
import cv2
import numpy as np
import math
from skimage.measure import compare_ssim as ssim
class TrainDataset(Dataset):
    def __init__(self):
        self.raw_path = opt.train_dataset_path
        self.tmp_path = opt.train_dataset_path
        self.disimg_gt_path = opt.train_dataset_path
        self.idx_txt = opt.gt_path
        self.raw_images = []
        self.tmp_images = []
        self.scores_gt=[]
        self.disimg_gt_images = []
        self.raw_names = []
        self.tmp_names = []
        self.disimg_gt_names = []
        with open(self.idx_txt) as f:
            for line in f:
                words = line.strip().split(',')
                self.raw_images.append(os.path.join(self.raw_path, words[0].split(':')[0][1:]))
                self.tmp_images.append(os.path.join(self.tmp_path, words[1].split(':')[0][1:]))
                self.scores_gt.append(float(words[3]))
                self.disimg_gt_images.append(os.path.join(self.disimg_gt_path, words[2]))
                self.raw_names.append(words[0].split(':')[0][1:])
                self.tmp_names.append(words[1].split(':')[0][1:])
                self.disimg_gt_names.append(words[2])

        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __getitem__(self, index):

        raw_image = self.transform(Image.open(self.raw_images[index]))
        tmp_image = self.transform(Image.open(self.tmp_images[index]))
        score_gt=self.scores_gt[index]
        disimg_gt_image =self.transform(Image.open(self.disimg_gt_images[index]))
        raw_names = self.raw_names[index]
        tmp_names = self.tmp_names[index]
        disimg_gt_names = self.disimg_gt_names[index]

        disimg_gt_train32 = np.load('./data/train/32/{}-{}-{}.npy'.format(raw_names[:-4], tmp_names[:-4],disimg_gt_names[:-4]))
        disimg_gt_train32 = torch.tensor(disimg_gt_train32)

        disimg_gt_train16 = np.load('./data/train/16/{}-{}-{}.npy'.format(raw_names[:-4], tmp_names[:-4],disimg_gt_names[:-4]))
        disimg_gt_train16 = torch.tensor(disimg_gt_train16)

        disimg_gt_train8 = np.load('./data/train/8/{}-{}-{}.npy'.format(raw_names[:-4], tmp_names[:-4],disimg_gt_names[:-4]))
        disimg_gt_train8 = torch.tensor(disimg_gt_train8)
        disimg_gt_train1 = np.load('./data/train/{}-{}-{}.npy'.format(raw_names[:-4], tmp_names[:-4], disimg_gt_names[:-4]))
        return raw_image,tmp_image, score_gt,disimg_gt_train32,disimg_gt_train16,disimg_gt_train8,disimg_gt_train1,raw_names,tmp_names,disimg_gt_names
    def __len__(self):
        return len(self.raw_images)
    

class EvalDataset(Dataset):
    def __init__(self):
        self.raw_path = opt.evl_dataset_path
        self.tmp_path = opt.evl_dataset_path
        self.disimg_gt_path = opt.evl_dataset_path
        self.idx_txt = opt.evl_gt_path
        self.raw_images = []
        self.tmp_images = []
        self.scores_gt=[]
        self.disimg_gt_images = []
        self.raw_names = []
        self.tmp_names = []
        self.disimg_gt_names = []
        with open(self.idx_txt) as f:
            for line in f:
                words = line.strip().split(',')
                self.raw_images.append(os.path.join(self.raw_path, words[0]))
                self.tmp_images.append(os.path.join(self.tmp_path, words[1]))
                self.scores_gt.append(float(words[2]))
                self.disimg_gt_images.append(os.path.join(self.disimg_gt_path, words[3]))
                self.raw_names.append(words[0])
                self.tmp_names.append(words[1])
                self.disimg_gt_names.append(words[3])

        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __getitem__(self, index):

        raw_image = self.transform(Image.open(self.raw_images[index]))
        tmp_image = self.transform(Image.open(self.tmp_images[index]))
        score_gt = self.scores_gt[index]
        disimg_gt_image = self.transform(Image.open(self.disimg_gt_images[index]))
        raw_names = self.raw_names[index]
        tmp_names = self.tmp_names[index]
        disimg_gt_names = self.disimg_gt_names[index]
        disimg_gt_train1 = np.load('/data/SUIM/disimg-ssim-gt1/{}-{}-{}.npy'.format(raw_names[:-4], tmp_names[:-4],disimg_gt_names[:-4]))
        disimg_gt_train1 = torch.tensor(disimg_gt_train1)
        return raw_image,tmp_image, score_gt,disimg_gt_train1, raw_names,tmp_names,disimg_gt_names
    def __len__(self):
        return len(self.raw_images)
    

class TestDataset(Dataset):
    def __init__(self):
        self.raw_path = opt.train_dataset_path
        self.tmp_path = opt.train_dataset_path
        self.disimg_gt_path = opt.train_dataset_path
        self.idx_txt ='/data/train_txt/test-easy.txt'
        self.raw_images = []
        self.tmp_images = []
        self.scores_gt=[]
        self.disimg_gt_images = []
        self.raw_names = []
        self.tmp_names = []
        self.disimg_gt_names = []
        with open(self.idx_txt) as f:
            for line in f:
                words = line.strip().split(',')
                self.raw_images.append(os.path.join(self.raw_path, words[0].split(':')[0][1:]))
                self.tmp_images.append(os.path.join(self.tmp_path, words[1].split(':')[0][1:]))
                self.scores_gt.append(float(words[3]))
                self.disimg_gt_images.append(os.path.join(self.disimg_gt_path, words[2]))
                self.raw_names.append(words[0].split(':')[0][1:])
                self.tmp_names.append(words[1].split(':')[0][1:])
                self.disimg_gt_names.append(words[2])

        self.transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    def __getitem__(self, index):
        raw_image = self.transform(Image.open(self.raw_images[index]))
        tmp_image = self.transform(Image.open(self.tmp_images[index]))
        score_gt=self.scores_gt[index]
        disimg_gt_image =self.transform(Image.open(self.disimg_gt_images[index]))
        raw_names = self.raw_names[index]
        tmp_names = self.tmp_names[index]
        disimg_gt_names = self.disimg_gt_names[index]
        disimg_gt_train1 = np.load('./data/wuli/QC-network/区分难度的模型/test-easy差异图真值/{}-{}-{}.npy'.format(raw_names[:-4], tmp_names[:-4],disimg_gt_names[:-4]))
        disimg_gt_train1 = torch.tensor(disimg_gt_train1)
        return raw_image,tmp_image, score_gt,disimg_gt_train1,raw_names,tmp_names,disimg_gt_names
    def __len__(self):
        return len(self.raw_images)




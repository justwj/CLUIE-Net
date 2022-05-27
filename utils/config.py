import torch
import math
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
import cv2 as cv
class Config:
    """
    Default configuration parameters
    """

    train_dataset_path = './data/E_img/'
    gt_path = './data/train_txt/train-easy.txt'


    # Training parameters
    batch_size = 1
    num_workers = 1
    lr_decay = 0.95
    if_lr_decay = False
    weight_decay = 0.01
    num_channels = 3
    style_loss_lambda = 1000
    re_loss_lambda = 1
    print_freq = 20
    isTrain = True

    model_save_path = './checkpoint'

    ifsave_result = False

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        :param kwargs:
        :return:
        """

        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                print("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

    def print_options(self):
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

    def Arrdistance(self,img1, img2):
        # Create a two-dimensional array to store the Euclidean distance of each pixel
        img1_r = img1[0, :, :]
        img1_g = img1[1, :, :]
        img1_b = img1[2, :, :]
        img2_r = img2[0, :, :]
        img2_g = img2[1, :, :]
        img2_b = img2[2, :, :]
        # Calculate the Euclidean distance between corresponding pixels
        dis = torch.sqrt(
            torch.pow((img1_r - img2_r), 2) + torch.pow((img1_g - img2_g), 2) + torch.pow((img1_b - img2_b), 2))
        # Normalize the Euclidean distance to [0,1]. The greater the Euclidean distance between two pixels, the greater the difference, the smaller the DIS, and the lighter the color in the heat map
        return dis

    def divide_method1(self,img, m, n):
        img = np.asarray(img)
        h, w = img.shape[0], img.shape[1]
        grid_h = int(h * 1.0 / (m - 1))
        grid_w = int(w * 1.0 / (n - 1))
        gx, gy = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, m))
        gx = gx.astype(np.int)
        gy = gy.astype(np.int)
        divide_image = np.zeros([m - 1, n - 1, grid_h, grid_w, 3],
                                np.uint8)
        for i in range(m - 1):
            for j in range(n - 1):
                divide_image[i, j, ...] = img[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1], :]
        return divide_image

    def divide_method(self,img, m, n):
        img = np.asarray(img)
        h, w = img.shape[0], img.shape[1]
        grid_h = int(h * 1.0 / m)
        grid_w = int(w * 1.0 / n)
        gx, gy = np.meshgrid(np.linspace(0, w, n + 1), np.linspace(0, h, m + 1))
        gx = gx.astype(np.int)
        gy = gy.astype(np.int)
        divide_image = np.zeros([m, n, grid_h, grid_w, 3],
                                np.uint8)
        for i in range(m):
            for j in range(n):
                divide_image[i, j] = img[gy[i][j]:gy[i + 1][j + 1], gx[i][j]:gx[i + 1][j + 1]]
        return divide_image

    # Calculate the true value of difference diagram according to SSIM
    def disimg_gt(self,img1, img2, ref):
        divide_image1_64 = Config.divide_method(self,img1, 4, 4)
        divide_image2_64 = Config.divide_method(self,img2, 4, 4)
        divide_image_ref_64 = Config.divide_method(self,ref, 4, 4)
        divide_image1_32 = Config.divide_method(self,img1, 8, 8)
        divide_image2_32 = Config.divide_method(self,img2, 8, 8)
        divide_image_ref_32 = Config.divide_method(self,ref, 8, 8)
        divide_image1_16 = Config.divide_method(self,img1, 16, 16)
        divide_image2_16 =Config.divide_method(self,img2, 16, 16)
        divide_image_ref_16 = Config.divide_method(self,ref, 16, 16)
        arr1_64 = np.zeros((16, 16))
        arr1_32 = np.zeros((16, 16))
        arr1_16 = np.zeros((16, 16))
        arr2_64 = np.zeros((16, 16))
        arr2_32 = np.zeros((16, 16))
        arr2_16 = np.zeros((16, 16))
        arr = np.zeros((16, 16))
        for i in range(4):
            for j in range(4):
                img1_ref_ssim_64 = ssim(divide_image1_64[i, j], divide_image_ref_64[i, j], multichannel=True)
                arr1_64[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = img1_ref_ssim_64
                img2_ref_ssim_64 = ssim(divide_image2_64[i, j], divide_image_ref_64[i, j], multichannel=True)
                arr2_64[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = img2_ref_ssim_64
        for i in range(8):
            for j in range(8):
                img1_ref_ssim_32 = ssim(divide_image1_32[i, j], divide_image_ref_32[i, j], multichannel=True)
                arr1_32[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = img1_ref_ssim_32
                img2_ref_ssim_32 = ssim(divide_image2_32[i, j], divide_image_ref_32[i, j], multichannel=True)
                arr2_32[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = img2_ref_ssim_32
        for i in range(16):
            for j in range(16):
                img1_ref_ssim_16 = ssim(divide_image1_16[i, j], divide_image_ref_16[i, j], multichannel=True)
                arr1_16[i, j] = img1_ref_ssim_16
                img2_ref_ssim_16 = ssim(divide_image2_16[i, j], divide_image_ref_16[i, j], multichannel=True)
                arr2_16[i, j] = img2_ref_ssim_16
        arr1 = arr1_64 * 0.2 + arr1_32 * 0.3 + arr1_16 * 0.5
        arr2 = arr2_64 * 0.2 + arr2_32 * 0.3 + arr2_16 * 0.5
        arr = arr1 - arr2
        zero = np.zeros_like(arr)
        disimg_gt_train1 = np.where(arr <= 0, zero, arr)
        disimg_gt_train2 = np.where(arr >= 0, zero, arr)
        disimg_gt_train1 = np.power(disimg_gt_train1, 0.5)
        disimg_gt_train2 = -np.power(-disimg_gt_train2, 0.5)
        disimg_gt_train = disimg_gt_train1 + disimg_gt_train2
        return disimg_gt_train
    #Calculate the true value of difference diagram according to MSE
    def disimg_gt_mse(self,img1, img2, ref):
        divide_image1_64 = Config.divide_method(self,img1, 4, 4)
        divide_image2_64 = Config.divide_method(self,img2, 4, 4)
        divide_image_ref_64 = Config.divide_method(self,ref, 4, 4)
        divide_image1_32 = Config.divide_method(self,img1, 8, 8)
        divide_image2_32 = Config.divide_method(self,img2, 8, 8)
        divide_image_ref_32 = Config.divide_method(self,ref, 8, 8)
        divide_image1_16 = Config.divide_method(self,img1, 16, 16)
        divide_image2_16 =Config.divide_method(self,img2, 16, 16)
        divide_image_ref_16 = Config.divide_method(self,ref, 16, 16)
        arr1_64 = np.zeros((16, 16))
        arr1_32 = np.zeros((16, 16))
        arr1_16 = np.zeros((16, 16))
        arr2_64 = np.zeros((16, 16))
        arr2_32 = np.zeros((16, 16))
        arr2_16 = np.zeros((16, 16))
        arr = np.zeros((16, 16))
        for i in range(4):
            for j in range(4):
                img1_ref_mse_64 = mse(divide_image1_64[i, j], divide_image_ref_64[i, j])
                arr1_64[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = img1_ref_mse_64
                img2_ref_mse_64 = mse(divide_image2_64[i, j], divide_image_ref_64[i, j])
                arr2_64[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = img2_ref_mse_64
        for i in range(8):
            for j in range(8):
                img1_ref_mse_32 = mse(divide_image1_32[i, j], divide_image_ref_32[i, j])
                arr1_32[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = img1_ref_mse_32
                img2_ref_mse_32 = mse(divide_image2_32[i, j], divide_image_ref_32[i, j])
                arr2_32[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = img2_ref_mse_32
        for i in range(16):
            for j in range(16):
                img1_ref_mse_16 = mse(divide_image1_16[i, j], divide_image_ref_16[i, j])
                arr1_16[i, j] = img1_ref_mse_16
                img2_ref_mse_16 = mse(divide_image2_16[i, j], divide_image_ref_16[i, j])
                arr2_16[i, j] = img2_ref_mse_16
        arr1 = arr1_64 * 0.2 + arr1_32 * 0.3 + arr1_16 * 0.5
        arr2 = arr2_64 * 0.2 + arr2_32 * 0.3 + arr2_16 * 0.5

        #Normalize MSE to [-1,1]
        # a=np.ones_like(arr1)
        # a=a*np.min(arr1)/(np.max(arr1)-np.min(arr1))
        # b=a*np.min(arr2)/(np.max(arr2)-np.min(arr2))
        # arr1=(arr1/(np.max(arr1)-np.min(arr1)))-a
        # arr2=(arr2/(np.max(arr2)-np.min(arr2)))-b


        arr = -(arr1 - arr2)/(255*255)
        zero = np.zeros_like(arr)
        disimg_gt_train1 = np.where(arr <= 0, zero, arr)
        disimg_gt_train2 = np.where(arr >= 0, zero, arr)
        disimg_gt_train1 = np.power(disimg_gt_train1, 0.1)
        disimg_gt_train2 = -np.power(-disimg_gt_train2, 0.1)
        disimg_gt_train = disimg_gt_train1 + disimg_gt_train2

        return disimg_gt_train



opt = Config()
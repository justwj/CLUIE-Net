from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import random

class traindataset(Dataset):
    def __init__(self, data_path, label_path, img_format='png'):
        self.data_path = data_path
        self.label_path = label_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.cl_images = []
        for img in self.uw_images:
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img).split('.')[0]+'fusion.png'))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            cl_img = self.transform(Image.open(self.cl_images[index]))
            name = os.path.basename(self.uw_images[index])
            return uw_img, cl_img,name
    def __len__(self):
        return len(self.uw_images)
    

class testdataset(Dataset):
    def __init__(self, data_path, img_format='png'):
        self.data_path = data_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            name = os.path.basename(self.uw_images[index])
            return uw_img,name
    def __len__(self):
        return len(self.uw_images)


class traindataset1(Dataset):
    def __init__(self, data_path, label_path, enhance_path, enhance_methods, img_format='.png'):
        self.data_path = data_path
        self.label_path = label_path
        self.enhance_path = enhance_path
        self.enhance_methods = enhance_methods
        self.uw_images = glob(os.path.join(self.data_path, '*' + img_format))
        self.cl_images=[]
        self.loss1_images=[]
        self.loss2_images = []
        self.loss3_images = []
        self.loss4_images = []
        self.loss5_images = []
        self.loss6_images = []
        self.loss7_images = []
        self.loss8_images = []
        self.loss9_images = []
        self.loss10_images = []
        self.loss11_images = []
        self.loss12_images = []
        self.loss13_images = []
        self.loss14_images = []
        for img in self.uw_images:
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img)))
            self.loss1_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[0] +img_format))
            self.loss2_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[1] + img_format))
            self.loss3_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[2] + img_format))
            self.loss4_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[3] + img_format))
            self.loss5_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[4] + img_format))
            self.loss6_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[5] + img_format))
            self.loss7_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[6] + img_format))
            self.loss8_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[7] + img_format))
            self.loss9_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[8] + img_format))
            self.loss10_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[9] + img_format))
            self.loss11_images.append(os.path.join(self.enhance_path, os.path.basename(img)[:-4] + enhance_methods[10] +img_format))

        self.transform = transforms.Compose([transforms.ToTensor()])
        self.transform1 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        uw_img = self.transform(Image.open(self.uw_images[index]))
        cl_img = self.transform(Image.open(self.cl_images[index]))
        loss1_img=self.transform1(Image.open(self.loss1_images[index]))
        loss2_img = self.transform1(Image.open(self.loss2_images[index]))
        loss3_img = self.transform1(Image.open(self.loss3_images[index]))
        loss4_img = self.transform1(Image.open(self.loss4_images[index]))
        loss5_img = self.transform1(Image.open(self.loss5_images[index]))
        loss6_img = self.transform1(Image.open(self.loss6_images[index]))
        loss7_img = self.transform1(Image.open(self.loss7_images[index]))
        loss8_img = self.transform1(Image.open(self.loss8_images[index]))
        loss9_img = self.transform1(Image.open(self.loss9_images[index]))
        loss10_img = self.transform1(Image.open(self.loss10_images[index]))
        loss11_img = self.transform1(Image.open(self.loss11_images[index]))

        uw_name = os.path.basename(self.uw_images[index])
        loss_name=os.path.basename(self.loss1_images[index])
        return uw_img,cl_img, loss1_img,loss2_img,loss3_img,loss4_img,loss5_img,loss6_img,loss7_img,loss8_img,loss9_img,loss10_img,loss11_img,uw_name,loss_name
    def __len__(self):
        return len(self.uw_images)
    
    
class traindataset2(Dataset):
    def __init__(self, data_path, label_path,enhance_path, img_format='jpg'):
        self.data_path = data_path
        self.label_path = label_path
        self.enhance_path=enhance_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.uw_images1=[]
        self.cl_images = []
        for img in self.uw_images:
            self.uw_images1.append(os.path.join(self.enhance_path, os.path.basename(img).split('.')[0])+'FUSION.jpg')
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img)))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            uw_img1 = self.transform(Image.open(self.uw_images1[index]))
            cl_img = self.transform(Image.open(self.cl_images[index]))
            name = os.path.basename(self.uw_images[index])
            return uw_img,uw_img1, cl_img,name
    def __len__(self):
        return len(self.uw_images)


class testdataset2(Dataset):
    def __init__(self, data_path, label_path,img_format='png'):
        self.data_path = data_path
        self.label_path = label_path
        self.uw_images = glob(os.path.join(self.data_path, '*.' + img_format))
        self.cl_images = []
        for img in self.uw_images:
            self.cl_images.append(os.path.join(self.label_path, os.path.basename(img)))
        self.transform = transforms.Compose([transforms.ToTensor()])
    def __getitem__(self, index):
            uw_img = self.transform(Image.open(self.uw_images[index]))
            cl_img = self.transform(Image.open(self.cl_images[index]))
            name = os.path.basename(self.uw_images[index])
            return uw_img, cl_img,name
    def __len__(self):
        return len(self.uw_images)
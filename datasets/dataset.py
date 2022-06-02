from torch.utils.data.dataset import Dataset
from glob import glob
import os
from PIL import Image
from torchvision import transforms
import random


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

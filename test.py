import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from datasets.dataset import testdataset
from tqdm import tqdm
import numpy as np
from models.networks import UNetEncoder, UNetDecoder
import click
import os
from torchvision import transforms
from PIL import Image
import time
import cv2


def write_to_log(log_file_path, status):
    with open(log_file_path, "a") as log_file:
        log_file.write(status + '\n')


def to_img(x,wide,hig):
    """Convert the tanh (-1 to 1) ranged tensor to image (0 to 1) tensor	"""
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, wide, hig)
    return x


def output(fE, fI, dataloader,output_path):
    fE.eval()
    fI.eval()
    for idx, data in tqdm(enumerate(dataloader)):
        inputtestimg, testname = data
        inputtestimg = Variable(inputtestimg).cuda()
        fE_out, enc_outs = fE(inputtestimg)
        fI_out = to_img(fI(fE_out, enc_outs),inputtestimg.shape[2],inputtestimg.shape[3])
        save_image(fI_out.cpu().data, output_path + '/{}'.format(testname[0]))


@click.command()
@click.argument('name', default='UIEBD')
@click.option('--test_path', default='./data/test_demo')
@click.option('--fe_load_path', default='./ckpt/fE_latest.pth')
@click.option('--fi_load_path', default='./ckpt/fI_latest.pth')
@click.option('--output_path', default='./output')


def main(name, test_path, fe_load_path, fi_load_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    fE_load_path = fe_load_path
    fI_load_path = fi_load_path
    test_dataset = testdataset(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    fE = UNetEncoder().cuda()
    fI = UNetDecoder().cuda()
    fE.load_state_dict(torch.load(fE_load_path))
    fI.load_state_dict(torch.load(fI_load_path))
    output(fE, fI, test_dataloader,output_path)

    
    
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()



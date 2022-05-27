import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from dataset.dataset import testdataset
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


def output(fE, fI, dataloader):
    fE.eval()
    fI.eval()
    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    for idx, data in tqdm(enumerate(dataloader)):

        inputtestimg, gttestimg, testname = data
        inputtestimg = Variable(inputtestimg).cuda()
        gttestimg = Variable(gttestimg, requires_grad=False).cuda()
        fE_out, enc_outs = fE(inputtestimg)
        fI_out = to_img(fI(fE_out, enc_outs),inputtestimg.shape[2],inputtestimg.shape[3])
        save_image(fI_out.cpu().data, 'CLUIE/{}'.format(testname[0]))
        fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        gttestimg = (gttestimg * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
        ssim_scores.append(ssim(fI_out, gttestimg, multichannel=True))
        psnr_scores.append(psnr(gttestimg, fI_out))
        mse_scores.append(mse(gttestimg, fI_out))
    return sum(ssim_scores) / len(dataloader), sum(psnr_scores) / len(dataloader), sum(mse_scores) / len(dataloader)
@click.command()
@click.argument('name', default='UIEBD')
@click.option('--test_path', default='/data/UIEBD/test')
@click.option('--gt_path', default='/data/UIEBD/gt')
@click.option('--fe_load_path', default='./ckpt/fE_latest.pth')
@click.option('--fi_load_path', default='./ckpt/fI_latest.pth')
def main(name, test_path, gt_path, fe_load_path, fi_load_path):
    fE_load_path = fe_load_path
    fI_load_path = fi_load_path
    test_dataset = testdataset2(test_path, gt_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    fE = UNetEncoder().cuda()
    fI = UNetDecoder().cuda()
    fE.load_state_dict(torch.load(fE_load_path))
    fI.load_state_dict(torch.load(fI_load_path))
    SSIM, PSNR, MSE = output(fE, fI, test_dataloader)
    test = ' test--ssim:{},psnr:{},mse:{}'.format( SSIM, PSNR, MSE)
    print(test)
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()



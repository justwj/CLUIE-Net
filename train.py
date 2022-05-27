#Piexl error loss(L1loss)+Content loss(MSEloss)+Superiority Discriminative loss(SDloss)
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from dataset.dataset import traindataset1,testdataset
from tqdm import tqdm
import numpy as np
import models
from models.networks import UNetEncoder, UNetDecoder
from models.net import ResNet50
import click
import os
from torchvision.models import vgg16
import cv2
from SDloss import SDLoss
import time
from torchvision import transforms
from PIL import Image
def to_img(x):
	x = 0.5 * (x + 1)
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 3, 256, 256)
	return x
def set_requires_grad(nets, requires_grad=False):
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad
	return requires_grad
def compute_val_metrics(fE, fI, dataloader):
	fE.eval()
	fI.eval()
	mse_scores = []
	ssim_scores = []
	psnr_scores = []
	for idx, data in tqdm(enumerate(dataloader)):
		inputtestimg, gttestimg, testname= data
		inputtestimg = Variable(inputtestimg).cuda()
		gttestimg = Variable(gttestimg, requires_grad=False).cuda()
		fE_out, enc_outs = fE(inputtestimg)
		fI_out = to_img(fI(fE_out, enc_outs))
		fI_out = (fI_out * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
		gttestimg = (gttestimg * 255).squeeze(0).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
		ssim_scores.append(ssim(fI_out, gttestimg, multichannel=True))
		psnr_scores.append(psnr(gttestimg, fI_out))
		mse_scores.append(mse(gttestimg, fI_out))
	fE.train()
	fI.train()
	return sum(ssim_scores)/len(dataloader), sum(psnr_scores)/len(dataloader), sum(mse_scores)/len(dataloader)
def backward_I_loss(fI, fE_out, enc_outs,gttrainimg,loss1img,loss2img,loss3img,loss4img,loss5img,loss6img,loss7img,loss8img,loss9img,loss10img,loss11img,criterion_L1,criterion_loss_content,vgg,criterion_sdloss,loss_net, optimizer_fI, lambda_I_loss, retain_graph):
	out=(fI(fE_out, enc_outs))
	fI_out = to_img(fI(fE_out, enc_outs))
	I_loss1 = criterion_L1(fI_out, gttrainimg)*lambda_I_loss#得loss
	I_loss2=criterion_loss_content(vgg(fI_out), vgg(gttrainimg))
	I_loss3=criterion_sdloss(loss_net,out,loss1img,loss2img,loss3img,loss4img,loss5img,loss6img,loss7img,loss8img,loss9img,loss10img,loss11img)*0.1
	I_loss=I_loss1+I_loss2+I_loss3
	optimizer_fI.zero_grad()
	I_loss.backward(retain_graph=retain_graph)
	optimizer_fI.step()
	return fI_out, I_loss,I_loss1,I_loss2,I_loss3 
def write_to_log(log_file_path, status):
	with open(log_file_path, "a") as log_file:
		log_file.write(status+'\n')
@click.command()
@click.argument('name',default='UIEBD')
@click.option('--train_path', default='./data/UIEBD/train')
@click.option('--test_path', default='./data/UIEBD/test')
@click.option('--gt_path', default='./data/UIEBD/gt')

@click.option('--gt1_path', default='./data/UIEBD/gt')
@click.option('--enhance_path', default='./data/UIEBD/e_results')
#@click.option('--enhance_methods', default=["CLAHE", "GCHE", "DIVE", "FUSION", "TWOSTEP","DCP", "HP","HUE", "IBLA", "RETINEX",  "UCM", "ULAP"])
@click.option('--enhance_methods', default=['MSCNN','UDCP','two_step','Regression-based','Retinex','fusion','dive+','DCP','GDCP','Histogram','Blurriness-based'], type=list)
@click.option('--batch_size', default=2)
@click.option('--learning_rate', default=1e-4)
@click.option('--start_epoch', default=1)
@click.option('--end_epoch', default=200)
@click.option('--save_interval', default=5)
@click.option('--fe_load_path', default='./ckpt/fE_latest.pth')
@click.option('--fi_load_path', default='./ckpt/fI_latest.pth')
@click.option('--lambda_i_loss', default=20.0)
@click.option('--continue_train', is_flag=False)
@click.option('--saveidx',default=10)
def main(name,train_path,test_path,gt_path,gt1_path,enhance_path,enhance_methods, learning_rate, batch_size, save_interval, start_epoch, end_epoch, fe_load_path, fi_load_path,  lambda_i_loss, continue_train,saveidx):
	fE_load_path = fe_load_path
	fI_load_path = fi_load_path
	lambda_I_loss = lambda_i_loss
	train_dataset = traindataset1(train_path,gt_path,enhance_path,enhance_methods)
	test_dataset = testdataset(test_path,gt1_path)
	train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
	fE = UNetEncoder().cuda()
	fI = UNetDecoder().cuda()
	criterion_L1 = nn.L1Loss().cuda()
	vgg = vgg16().cuda()
	state_dict = torch.load('./data/vgg/vgg16-397923af.pth')
	vgg.load_state_dict(state_dict)
	vgg.eval()
	set_requires_grad(vgg, False)
	criterion_loss_content = nn.MSELoss().cuda()
	loss_net = ResNet50().cuda()
	loss_net.load_state_dict(torch.load(os.path.join('./data/QC_ckpt/net_idx12.pkl')))
	loss_net.eval()
	set_requires_grad(loss_net, False)
	criterion_sdloss=SDLoss().cuda()

	optimizer_fE = torch.optim.Adam(fE.parameters(), lr=learning_rate)
	optimizer_fI = torch.optim.Adam(fI.parameters(), lr=learning_rate)
	fE.train()
	fI.train()
	log_file_path = './ckpt/log_file.txt'
	log_file_path1 = './ckpt/log_file_test.txt'
	if continue_train:
		if fE_load_path:
			fE.load_state_dict(torch.load(fE_load_path))
			print('Loaded fE from {}'.format(fE_load_path))
		if fI_load_path:
			fI.load_state_dict(torch.load(fI_load_path))
			print('Loaded fI from {}'.format(fI_load_path))
#train
	for epoch in range(start_epoch, end_epoch):
		for idx, data in tqdm(enumerate(train_dataloader)):
			inputtrainimg, gttrainimg,loss1img,loss2img,loss3img,loss4img,loss5img,loss6img,loss7img,loss8img,loss9img,loss10img,loss11img,train_names,loss_names= data
			
			inputtrainimg = Variable(inputtrainimg).cuda()
			gttrainimg = Variable(gttrainimg, requires_grad=False).cuda()
			loss1img=Variable(loss1img).cuda()
			loss2img=Variable(loss2img).cuda()
			loss3img=Variable(loss3img).cuda()
			loss4img = Variable(loss4img).cuda()
			loss5img = Variable(loss5img).cuda()
			loss6img = Variable(loss6img).cuda()
			loss7img = Variable(loss7img).cuda()
			loss8img = Variable(loss8img).cuda()
			loss9img = Variable(loss9img).cuda()
			loss10img = Variable(loss10img).cuda()
			loss11img = Variable(loss11img).cuda()
			fE_out, enc_outs = fE(inputtrainimg)
			optimizer_fE.zero_grad()
			fI_out, I_loss,I_loss1,I_loss2,I_loss3 = backward_I_loss(fI, fE_out, enc_outs, gttrainimg,loss1img,loss2img,loss3img,loss4img,loss5img,loss6img,loss7img,loss8img,loss9img,loss10img,loss11img,criterion_L1,criterion_loss_content,vgg,criterion_sdloss,loss_net, optimizer_fI, lambda_I_loss, retain_graph=False)
			progress = "Epoch: {},Iter: {},I_loss: {},I_loss1: {},I_loss2: {},I_loss3:{}".format(epoch, idx, I_loss.item(),I_loss1.item(), I_loss2.item(),I_loss3.item())
			optimizer_fE.step()

			if idx % saveidx == 0:
				print(progress)
				write_to_log(log_file_path, progress)

#save model
		torch.save(fE.state_dict(), './ckpt/fE_latest.pth')
		torch.save(fI.state_dict(), './ckpt/fI_latest.pth')
		if epoch % save_interval == 0:
			torch.save(fE.state_dict(), './ckpt/fE_{}.pth'.format(epoch))
			torch.save(fI.state_dict(), './ckpt/fI_{}.pth'.format(epoch))
			# val
		ssim, psnr, mse = compute_val_metrics(fE, fI, test_dataloader)
		test = '{}epoch test--ssim:{},psnr:{},mse:{}'.format(epoch, ssim, psnr, mse)
		print(test)
		write_to_log(log_file_path1, test)
	status = 'End of epoch. Models saved.'
	print(status)
if __name__== "__main__":
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	main()



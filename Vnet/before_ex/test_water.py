import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import Dataset

import os
from before_ex import vnet_only_w_test

import time
#import matplotlib.pyplot as plt
#import cv2
from skimage import io, transform

def dice_loss(pred, target):
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1-((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def L1_loss(pred, target):
    f_pred = pred.contiguous().view(-1)
    f_target = target.contiguous().view(-1)
    L1_loss_func = nn.L1Loss()
    return L1_loss_func(f_pred,f_target)


class Test_Water_Dataset(Dataset):
    def __init__(self, groundtruthdir, videodir, frm_ch_num=16, frm_period =5):
        self.gtdir = groundtruthdir
        self.videodir = videodir
        self.transform = transform
        self.videolist = sorted(os.listdir(videodir))
        self.gtlist = sorted(os.listdir(groundtruthdir))
        self.f_ch_num = frm_ch_num
        self.f_period = frm_period

    def __len__(self):
        return len(os.listdir(self.videodir))

    def __getitem__(self, idx):
        videoname = self.videolist[idx]
        frmspath = self.videodir + videoname + '/'
        frmsname = sorted(os.listdir(frmspath))

        frms = io.imread(frmspath + frmsname[0])# first frame
        gt = io.imread(self.gtdir + self.gtlist[idx])
        gt = gt[:, :, 0:3]

        height = frms.shape[0]
        width = frms.shape[1]

        gt_w = gt==255 #water groundtruth [0, 1]

        r_frms = np.reshape(frms[:, :, 0], (height, width, 1, 1))
        g_frms = np.reshape(frms[:, :, 1], (height, width, 1, 1))
        b_frms = np.reshape(frms[:, :, 2], (height, width, 1, 1))
        gts_w = gt_w.copy()


        for num in range(self.f_period, self.f_period*self.f_ch_num, self.f_period):#frame period in video
            frm = io.imread(frmspath + frmsname[num])

            r_frm = np.reshape(frm[:, :, 0], (height, width, 1, 1))
            g_frm = np.reshape(frm[:, :, 1], (height, width, 1, 1))
            b_frm = np.reshape(frm[:, :, 2], (height, width, 1, 1))

            r_frms = np.concatenate((r_frms, r_frm), axis=2)
            g_frms = np.concatenate((g_frms, g_frm), axis=2)
            b_frms = np.concatenate((b_frms, b_frm), axis=2)

            gts_w = np.concatenate((gts_w, gt_w), axis=2)

        frms = np.concatenate((r_frms, g_frms, b_frms), axis=3)

        gts_w2 = np.reshape(gts_w[:,:,0:self.f_ch_num], (height, width, self.f_ch_num, 1))

        frms = torch.tensor(frms, dtype=torch.float)
        gts_w2 = torch.tensor(gts_w2, dtype=torch.float)

        frms = frms.permute(3, 2, 0, 1)
        gts_w2 = gts_w2.permute(3, 2, 0, 1)

        frms = frms / 255

        return frms, gts_w2

print("="*20)
print("Check before Testing")
gtdir = "/home/hyeongeun/dataset/Test/annot/" #"/datahdd/dataset/water_segmentation/Train/annot/"
videodir = "/home/hyeongeun/dataset/Test/frames/" #"/datahdd/dataset/water_segmentation/Train/frames/"
print("Ground truth image dir : ", gtdir)
print("Frames dir : ", videodir)

print("-"*10)
print("Create Testset class")
frm_ch_num = 16
frm_period = 5
print("# of frames : ", frm_ch_num)
print("frame period : ", frm_period)
testset = Test_Water_Dataset(gtdir,videodir,frm_ch_num,frm_period)
print("Testset length : ",testset.__len__())
frms, gt_w = testset[0] #,gt_r
data_height = frms.shape[2]
data_width = frms.shape[3]
print("Frms shape : ",frms.shape)
print("Water ground truth shape : ", gt_w.shape)

print("-"*10)
print("Create Testset dataloader")
batch_sz = 1
print("Batch size : ",batch_sz)
testloaders =torch.utils.data.DataLoader(testset, batch_size = batch_sz, shuffle= False)
videos, gts_w = next(iter(testloaders)) #, gts_r , s_r
print("Videos shape : ",videos.shape)
print("Water gts shape : ",gts_w.shape)

print("-"*10)
print("Create model")
model_load_path="/home/hyeongeun/PycharmProjects/vnet/model_save/vnet_1ch_period_5_cyLR_batch8_diceL1_randcropflipnoise_191022_0.pth"
model= vnet_only_w_test.VNet(elu=False, nll=False, frm_ch=frm_ch_num, height=data_height, width=data_width)
model.load_state_dict(torch.load(model_load_path))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = False
if device.type == "cuda":
    model.to(device)
    use_gpu = True
print("Device type:", device.type)

model.test()

print("="*20)
print("Start test")
epoch_start_time =time.time()
with torch.no_grad():
    dice_loss = 0.0
    l1_loss = 0.0
    for frms, gts_w in testloaders:
        if use_gpu:
            frms, gts_w = frms.cuda(), gts_w.cuda()
        frms, gts_w = Variable(frms), Variable(gts_w)
        pred_water = model(frms)

        dice_loss += dice_loss(pred_water, gts_w)
        l1_loss += L1_loss(pred_water, gts_w)

        img_pred = pred_water.copy().view(-1, frm_ch_num, data_height, data_width)

        


    dice_loss /= len(testloaders)
    l1_loss /= len(testloaders)

    print("Test result : Dice loss = {:.5f}, L1 loss = {:.5f}".format(dice_loss,l1_loss))

epoch_time = time.time() - epoch_start_time
#time_h = int((epoch_time // 60) // 60)
time_m = int((epoch_time // 60) % 60)
time_s = epoch_time % 60
print("Time : {:.0f}min {:.1f}sec".format(time_m,time_s))
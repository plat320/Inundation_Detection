import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.autograd import Variable

from torch.utils.data import Dataset
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter

import os
from before_ex import vnet_only_w_test

import time
#import matplotlib.pyplot as plt
#import cv2
from skimage import io


def np_select_rand_pos(img):
    max_x = img.shape[1]-128
    max_y = img.shape[0]-128
    if max_x<0 or max_y<0:
        print("This image size is too small.")
        return 0, 0
    return np.random.randint(max_x), np.random.randint(max_y)

def np_crop_img(img,x,y):
    return img[y:y+128,x:x+128,:]

def np_rand_flip(img,flip_flag):
    if flip_flag:
        return np.flip(img,1)
    else:
        return img

def np_rand_noise(img):
    noise_flag = np.random.randint(2)
    if noise_flag:
        s = np.random.normal(0, 25, (128, 128, 3))
        tmp = img + s
        tmp[tmp>255] = 255
        tmp[tmp<0] = 0
        tmp = tmp.astype(np.uint8)
        return tmp
    else :
        return img


class CCTVDataset(Dataset):
    def __init__(self, groundtruthdir, videodir, transform=None, frm_ch_num=16, frm_period =5):
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
        flip_flag = np.random.randint(2)

        frms = io.imread(frmspath + frmsname[0])# first frame

        gt = io.imread(self.gtdir + self.gtlist[idx])
        gt = gt[:,:,0:3]
        #print(idx,",",videoname,", gt.shape : ",gt.shape)

        rand_x, rand_y = np_select_rand_pos(frms)

        frms = np_crop_img(frms, rand_x, rand_y)
        frms = np_rand_noise(frms)
        frms = np_rand_flip(frms,flip_flag)

        gt = np_crop_img(gt, rand_x, rand_y)
        gt = np_rand_flip(gt,flip_flag)

        gt_w = gt==255 #water groundtruth [0, 1]

        show_frms = torch.tensor(frms.copy(), dtype=torch.float)
        show_gt_w = torch.tensor(gt_w.copy(), dtype=torch.float)

        r_frms = np.reshape(frms[:, :, 0], (128, 128, 1, 1))
        g_frms = np.reshape(frms[:, :, 1], (128, 128, 1, 1))
        b_frms = np.reshape(frms[:, :, 2], (128, 128, 1, 1))
        gts_w = gt_w.copy()


        for num in range(self.f_period, self.f_period*self.f_ch_num, self.f_period):#frame period in video
            frm = io.imread(frmspath + frmsname[num])
            frm = np_crop_img(frm, rand_x, rand_y)
            frm = np_rand_noise(frm)
            frm = np_rand_flip(frm,flip_flag)

            r_frm = np.reshape(frm[:, :, 0], (128, 128, 1, 1))
            g_frm = np.reshape(frm[:, :, 1], (128, 128, 1, 1))
            b_frm = np.reshape(frm[:, :, 2], (128, 128, 1, 1))

            r_frms = np.concatenate((r_frms, r_frm), axis=2)
            g_frms = np.concatenate((g_frms, g_frm), axis=2)
            b_frms = np.concatenate((b_frms, b_frm), axis=2)

            gts_w = np.concatenate((gts_w, gt_w), axis=2)

        frms = np.concatenate((r_frms, g_frms, b_frms), axis=3)

        gt_w2 = np.reshape(gts_w[:,:,0:self.f_ch_num], (128, 128, self.f_ch_num, 1))
        #gt_w2 = np.reshape(gt_w2[:,:,:,0],(128,128,self.f_ch_num,1))

        frms = torch.tensor(frms, dtype=torch.float)
        gt_w2 = torch.tensor(gt_w2, dtype=torch.float)

        frms = frms.permute(3, 2, 0, 1)
        gt_w2 = gt_w2.permute(3, 2, 0, 1)

        frms = frms / 255

        return frms, gt_w2, show_frms, show_gt_w

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

def dice_focal_loss(pred, target, batch_size, gamma=2):
    f_pred = pred.contiguous().view(batch_size, -1)
    f_target = target.contiguous().view(batch_size, -1)

    gt1_mask = f_target.contiguous()
    gt0_mask = f_target == 0

    pt_gt1 = f_pred * gt1_mask
    pt_gt0 = 1. * gt0_mask - f_pred * gt0_mask
    pt = pt_gt1 + pt_gt0
    pt = torch.sum(pt, 1) / f_target.shape[1]

    smooth = 1.
    inter = torch.sum(f_pred*f_target,1)
    p_sum = torch.sum(f_pred*f_pred,1)
    g_sum = torch.sum(f_target * f_target, 1)
    dice = 1-((2.*inter + smooth)/(p_sum + g_sum + smooth))

    dice_focal = ((1-pt)**gamma)*dice
    dice_focal = dice_focal.sum()/batch_size

    return dice_focal


def diceL1_focal_loss(pred, target, batch_size, gamma=2):
    f_pred = pred.contiguous().view(batch_size, -1)
    #print("p ", f_pred)
    f_target = target.contiguous().view(batch_size, -1)
    #print("t ", f_target)
    gt1_mask = f_target.contiguous()
    gt0_mask = f_target == 0

    pt_gt1 = f_pred * gt1_mask
    pt_gt0 = 1. * gt0_mask - f_pred * gt0_mask
    pt = pt_gt1 + pt_gt0
    #print("pt ", pt)
    pt = torch.sum(pt, 1) / f_target.shape[1]

    #print(f_target.shape[1])
    #print("pt ", pt)
    smooth = 1.
    inter = torch.sum(f_pred * f_target, 1)
    p_sum = torch.sum(f_pred * f_pred, 1)
    g_sum = torch.sum(f_target * f_target, 1)
    dice = 1 - ((2. * inter + smooth) / (p_sum + g_sum + smooth))
    #print("dice ", dice)
    L1 = 1 - pt
    #print("L1 ", L1)
    diceL1_focal = ((1 - pt) ** gamma) * (dice + L1)
    #print("diceL1_focal ", diceL1_focal)
    diceL1_focal = diceL1_focal.sum() / batch_size

    return diceL1_focal
#--------------------
#all of parameter setting
set_gtdir = "/datahdd/dataset/water_segmentation/Train/annot/"#"/home/hyeongeun/dataset/Train/annot/"
set_videodir = "/datahdd/dataset/water_segmentation/Train/frames/"#"/home/hyeongeun/dataset/Train/frames/"
set_frm_ch_num = 16
set_frm_period = 5
set_batch_size = 8

set_base_lr = 0.0005 #scheduler setting
set_max_lr=0.01 #scheduler setting
set_step_size_up = 300 #scheduler setting

set_wt_save_path = '/datahdd/code/water detection/vnet.pytorch/model_save/' #'/home/hyeongeun/PycharmProjects/vnet/model_save/'
set_wt_save_name = 'vnet_FocaldiceL1_period_5_cyLR_batch8_randcropflipnoise_191026_0'
#--------------------
writer = SummaryWriter()
print("="*20)
print("Check before training")
gtdir =  set_gtdir
videodir = set_videodir
print("Ground truth image dir : ", gtdir)
print("Frames dir : ", videodir)

print("-"*10)
print("Create dataset class")

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128,128),0)
])

frm_ch_num = set_frm_ch_num
frm_period = set_frm_period
print("# of frames : ", frm_ch_num)
print("frame period : ", frm_period)
cctv_dataset = CCTVDataset(gtdir,videodir,transform,frm_ch_num,frm_period)
print("Dataset length : ",cctv_dataset.__len__())
frms, gt_w, _ , _ = cctv_dataset[0] #,gt_r
print("Frms shape : ",frms.shape)
print("Water ground truth shape : ", gt_w.shape)
data_height = frms.shape[2]
data_width = frms.shape[3]

print("-"*10)
print("Create dataloader")
batch_sz = set_batch_size
print("Batch size : ",batch_sz)
dataloaders =torch.utils.data.DataLoader(cctv_dataset, batch_size = batch_sz, shuffle= True)
videos, gts_w, s_v, s_w  = next(iter(dataloaders)) #, gts_r , s_r
print("Videos shape : ",videos.shape)
print("Water gts shape : ",gts_w.shape)
print("test v w : ", s_v.shape, s_w.shape) #, s_r.shape)

print("-"*10)
print("Create model")
model= vnet_only_w_test.VNet(elu=False, nll=False, frm_ch=frm_ch_num, height=data_height, width=data_width)
weight_decay = 1e-4
print("Weight decay : ",weight_decay)
optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr = set_base_lr, max_lr=set_max_lr,
                                  step_size_up = set_step_size_up, cycle_momentum=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = False
if device.type == "cuda":
    model.to(device)
    use_gpu = True
print("Device type:", device.type)

model.train()
epochs = 10000
best_epoch_loss = 1.0
best_epoch_loss_w = 1.0
best_epoch_loss_l1w = 1.0

best_water_loss = 1.0

print("="*20)
print("Start train")
for epoch in range(epochs):
    epoch_start_time =time.time()
    epoch_loss = 0.0
    water_epoch_loss = 0.0
    water_epoch_l1_loss = 0.0

    show10_video = None
    show10_w_gt = None
    show10_w_pred = None

    show20_video = None
    show20_w_gt = None
    show20_w_pred = None

    for batch_idx, (frms, gts_w, show_frms, show_gts_w) in enumerate(dataloaders): #, gts_r, show_gts_r
        #print("batch idx : ",batch_idx)
        if use_gpu:
            frms, gts_w = frms.cuda(), gts_w.cuda()#, gts_r.cuda()#, gts_r
        frms, gts_w = Variable(frms), Variable(gts_w)#, gts_r, Variable(gts_r)
        optimizer.zero_grad()
        #print(gts_w.shape)
        output = model(frms)
        #print(output.shape)
        pred_water = output #[:, :, 0]
        #print(batch_idx," : ",pred_water.max(),pred_water.min(),gts_w.max(),gts_w.min())

        #water_loss = dice_loss(pred_water, gts_w)
        #water_f_loss = mean_focal_loss(pred_water,gts_w)
        water_loss = diceL1_focal_loss(pred_water,gts_w,batch_sz)
        water_L1_loss = L1_loss(pred_water,gts_w)#water_f_loss
        loss = water_loss #L1_weight*(water_L1_loss) #+ road_loss + road_L1_loss)

        loss.backward()
        optimizer.step()

        epoch_loss += loss
        water_epoch_loss += water_loss
        water_epoch_l1_loss += water_L1_loss

        if batch_idx == 10:
            show10_w_pred = pred_water

            show10_w_gt = show_gts_w.permute(0, 3, 1, 2)
            #print("=" * 30)
            #print("show10_w_gt", show10_w_gt.shape)
            #print("max", show10_w_gt.max())
            #print("min", show10_w_gt.min())
            #print("=" * 30)
            show10_w_gt = show10_w_gt * 255

            show10_video = show_frms.permute(0, 3, 1, 2)

            show10_w_pred = show10_w_pred.view(-1,frm_ch_num, 128, 128)
            show10_w_pred = show10_w_pred[:, 0:3, :, :]
            #print("="*30)
            #print("show10_w_pred",show10_w_pred.shape)
            #print("max", show10_w_pred.max())
            #print("min", show10_w_pred.min())
            #print("=" * 30)

        if batch_idx == 20:
            show20_w_pred = pred_water

            show20_w_gt = show_gts_w.permute(0, 3, 1, 2)
            show20_w_gt = show20_w_gt * 255

            show20_video = show_frms.permute(0, 3, 1, 2)

            show20_w_pred = show20_w_pred.view(-1, frm_ch_num, 128, 128)
            show20_w_pred = show20_w_pred[:, 0:3, :, :]

    epoch_loss /= len(dataloaders)
    water_epoch_loss /= len(dataloaders)
    water_epoch_l1_loss /= len(dataloaders)

    if best_epoch_loss>epoch_loss:
        best_epoch_loss = epoch_loss
        best_epoch_loss_w = water_epoch_loss
        best_epoch_loss_l1w = water_epoch_l1_loss

        wt_save_path = set_wt_save_path
        wt_save_name = set_wt_save_name + '_best.pth'
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'best_epoch_loss':best_epoch_loss,
            'water_epoch_loss':water_epoch_loss
        }, wt_save_path+wt_save_name)

    wt_save_path = set_wt_save_path
    wt_save_name = set_wt_save_name +'.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_epoch_loss': best_epoch_loss,
        'water_epoch_loss': water_epoch_loss
    }, wt_save_path + wt_save_name)


    if best_water_loss>water_epoch_loss:
        best_water_loss = water_epoch_loss
    epoch_time = time.time() - epoch_start_time
    time_h = int((epoch_time // 60) // 60)
    time_m = int((epoch_time // 60) % 60)
    time_s = epoch_time % 60
    #print(time_h, "h ", time_m, "m ", time_s, "s")

    print("Epoch {}/{} Total loss : {:.8f} ( Best total loss : {:.8f} = dice_w {:.8f} + L1_w {:.8f} , lr = {} )".format(epoch, epochs - 1, epoch_loss, best_epoch_loss,
                                                                                                                        best_epoch_loss_w, best_epoch_loss_l1w,
                                                                                                                        scheduler.get_lr()))
    print("Water dice loss : {:.8f} ( Best water dice loss : {:.8f} )  /  Water L1 loss : {:.8f} / Time : {:.0f}min {:.1f}sec".format(water_epoch_loss, best_water_loss,water_epoch_l1_loss,
                                                                                                                         time_m,time_s))

    scheduler.step()
    writer.add_scalar('total loss/train', epoch_loss, epoch)
    writer.add_scalar('water dice loss/train', water_epoch_loss, epoch)
    writer.add_scalar('water L1 loss/train', water_epoch_l1_loss, epoch)

    grid_10video = utils.make_grid(show10_video,nrow=4,normalize=True)
    grid_10_w_gt = utils.make_grid(show10_w_gt, nrow=4,normalize=True)
    grid_10_w_pred = utils.make_grid(show10_w_pred, nrow=4,normalize=True)
    grid_10_w_pred_thres = grid_10_w_pred > 0.5

    grid_20video = utils.make_grid(show20_video,nrow=4,normalize=True)
    grid_20_w_gt = utils.make_grid(show20_w_gt,nrow=4,normalize=True)
    grid_20_w_pred = utils.make_grid(show20_w_pred,nrow=4,normalize=True)
    grid_20_w_pred_thres = grid_20_w_pred > 0.5

    writer.add_image('grid_10_video',grid_10video, epoch)
    writer.add_image('grid_10_water_gt', grid_10_w_gt, epoch)
    writer.add_image('grid_10_water_pred', grid_10_w_pred, epoch)
    writer.add_image('grid_10_water_pred_thres', grid_10_w_pred_thres, epoch)

    writer.add_image('grid_20_video', grid_20video, epoch)
    writer.add_image('grid_20_water_gt', grid_20_w_gt, epoch)
    writer.add_image('grid_20_water_pred', grid_20_w_pred, epoch)
    writer.add_image('grid_20_water_pred_thres', grid_20_w_pred_thres, epoch)



    ##----------------------------------------------end of each epoch------------------------------------------------------------------





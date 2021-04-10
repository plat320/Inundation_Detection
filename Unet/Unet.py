import time
import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import os
import torch.optim as optim
import torch.nn.functional as F
import cv2
import test

import unet_2D3D

def iou_cal(pred, target):
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return intersection / (A_sum + B_sum - intersection)

def val_3Dunet(model, optimizer, scheduler, valloader,
                     epoch, set_wt_save_path, set_wt_save_name,
                     best_mIoU, device):
    epoch_start_time = time.time()
    print("=" * 20)
    print("<start validation>")
    model.to(device)
    model.eval()

    with torch.no_grad():
        Total_w_mIoU = 0.0
        Total_r_mIoU = 0.0
        for idx, (frms, gts_w, gts_r) in enumerate(valloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)

            output = model(frms)

            water_pred = (output[:, 0, :, :, :] > 0.5) * 1
            road_pred = (output[:, 1, :, :, :] > 0.5) * 1

            w_mIoU = iou_cal(water_pred, gts_w)
            r_mIoU = iou_cal(road_pred, gts_r)

            Total_w_mIoU += w_mIoU
            Total_r_mIoU += r_mIoU

        Total_w_mIoU /= len(valloader)
        Total_r_mIoU /= len(valloader)
        print("Val result : Total_w_mIoU = {:.5f}, Total_r_mIoU = {:.5f}".format(Total_w_mIoU, Total_r_mIoU))

        if ( best_mIoU < (Total_w_mIoU+Total_r_mIoU)/2 ) :
            best_mIoU = (Total_w_mIoU+Total_r_mIoU)/2
            wt_save_path = set_wt_save_path
            wt_save_name = set_wt_save_name + '_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mIoU': best_mIoU,
                'best_w_mIoU': Total_w_mIoU,
                'best_r_mIoU': Total_r_mIoU
            }, wt_save_path + wt_save_name)
            print('**** Save new best.pth ****')
            print("Best mIoU = {:.5f}".format(best_mIoU))
            epoch_time = time.time() - epoch_start_time
            time_m = int((epoch_time // 60) % 60)
            time_s = epoch_time % 60
            print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
            print("=" * 20)
            return best_mIoU
        else :
            print("Best mIoU = {:.5f}".format(best_mIoU))
            epoch_time = time.time() - epoch_start_time
            time_m = int((epoch_time // 60) % 60)
            time_s = epoch_time % 60
            print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
            print("=" * 20)
            return best_mIoU

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


def L1_loss(pred, target):
    f_pred = pred.contiguous().view(-1)
    f_target = target.contiguous().view(-1)
    l1loss = (torch.abs(f_pred - f_target).sum()) / f_pred.shape[0]
    return l1loss

def train_3Dunet(model,optimizer,scheduler,trainloader,valloader,
           set_wt_save_path,set_wt_save_name,device,
           best_mIoU,start_epoch,num_epochs):
    best_epoch_loss = 4.0
    model.to(device)
    model.train()
    for epoch in range(start_epoch,num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        water_epoch_loss = 0.0
        road_epoch_loss = 0.0
        printloss = 0.0

        if epoch % 5 == 4 :
            wt_save_path = set_wt_save_path
            wt_save_name = set_wt_save_name + '_last.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, wt_save_path + wt_save_name)

            best_mIoU = val_3Dunet(model, optimizer, scheduler, valloader,
                     epoch, set_wt_save_path, set_wt_save_name,
                     best_mIoU, device)

        for batch_idx, (frms, gts_w, gts_r) in enumerate(trainloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)  # , gts_r.cuda()#, gts_r
            optimizer.zero_grad()
            output = model(frms)

            w_di = dice_loss(output[:, 0, :, :], gts_w)
            w_l1 = L1_loss(output[:, 0, :, :], gts_w)
            water_loss = (w_di + w_l1)

            r_di = dice_loss(output[:, 1, :, :], gts_r)
            r_l1 = L1_loss(output[:, 1, :, :], gts_r)
            road_loss = (r_di + r_l1)

            loss = water_loss + road_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss
            printloss += loss
            water_epoch_loss += water_loss
            road_epoch_loss += road_loss

            if batch_idx%10 == 9:
                printloss = printloss/20
                print('[%d, %5d] loss : %.5f' %(epoch, batch_idx+1, printloss))
                printloss = 0.0

        epoch_loss /= len(trainloader)
        water_epoch_loss /= len(trainloader)
        road_epoch_loss /= len(trainloader)


        summary.add_scalar('loss/epoch_loss', epoch_loss, epoch)
        summary.add_scalar('loss/water_epoch_loss', water_epoch_loss, epoch)
        summary.add_scalar('loss/road_epoch_loss', road_epoch_loss, epoch)

        epoch_time = time.time() - epoch_start_time

        time_m = int((epoch_time // 60) % 60)
        time_s = epoch_time % 60

        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch_loss_w = water_epoch_loss
            best_epoch_loss_r = road_epoch_loss

        print(
            "Epoch {}/{} Total loss : {:.8f} ( Best total loss : {:.8f} = dice_w {:.8f} + dice_r {:.8f} , lr = {} ,time = {:.0f}m {:.0f}s )".format(
                epoch, num_epochs - 1, epoch_loss, best_epoch_loss,
                best_epoch_loss_w, best_epoch_loss_r,
                scheduler.get_lr(),
                time_m, time_s))
        scheduler.step()

class Dataset3D(Dataset):
    def __init__(self, groundtruthdir, videodir, frm_ch_num=16, frm_period =5):
        self.gtdir = groundtruthdir
        self.videodir = videodir
        self.videolist = sorted(os.listdir(videodir))
        self.gtlist = sorted(os.listdir(groundtruthdir))
        self.f_ch_num = frm_ch_num
        self.f_period = frm_period
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             ])
        self.totnsr = transforms.Compose([transforms.ToTensor(),
                                          ])

    def __len__(self):
        return len(os.listdir(self.videodir))

    def __getitem__(self, idx):
        videoname = self.videolist[idx]
        frmspath = self.videodir + videoname + '/'
        frmsname = sorted(os.listdir(frmspath))

        height = 5*16*4
        width = 9*16*4

        frms = cv2.imread(frmspath + frmsname[0])# first frame

        color = frms.shape[2]
        frms = np.reshape(frms, (height, width, color, 1))

        for num in range(self.f_period, self.f_period*self.f_ch_num, self.f_period):#frame period in video
            frm = cv2.imread(frmspath + frmsname[num])
            frm = np.reshape(frm, (height, width, color, 1))
            frms = np.concatenate((frms, frm), axis=3)

        gt = cv2.imread(self.gtdir + self.gtlist[idx],0)

        gt = np.reshape(gt, (height, width, 1, 1))

        gt = np.concatenate((gt, gt, gt, gt), axis=3)  # [HWC4]
        gt = np.concatenate((gt, gt, gt, gt), axis=3)  # [HWC16]

        gt_w = (gt==255)*1.0 #water groundtruth [0, 1]
        gt_r = (gt==125)*1.0

        frms = torch.tensor(frms, dtype=torch.float)
        gts_w = torch.tensor(gt_w, dtype=torch.float)
        gts_r = torch.tensor(gt_r, dtype=torch.float)

        frms = frms.permute(2, 3, 0, 1)  # H W C F => C F H W
        gts_w = gts_w.permute(2, 3, 0, 1)
        gts_r = gts_r.permute(2, 3, 0, 1)

        frms = frms / 255

        return frms, gts_w, gts_r



class Dataset2D(Dataset):
    def __init__(self, groundtruthdir, imagedir):
        self.gtdir = groundtruthdir
        self.imgdir = imagedir
        self.imglist = sorted(os.listdir(imagedir))
        self.gtlist = sorted(os.listdir(groundtruthdir))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225]),
                                             ])
        self.totnsr = transforms.Compose([transforms.ToTensor(),
                                          ])

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgdir + self.imglist[idx])
        gt = cv2.imread(self.gtdir + self.gtlist[idx], cv2.IMREAD_GRAYSCALE)

        gt_w = (gt == 255) * 1.0
        gt_r = (gt == 125) * 1.0

        img = self.transform(img)
        gt_w = self.totnsr(gt_w)
        gt_r = self.totnsr(gt_r)

        return img, gt_w, gt_r

if __name__ == '__main__':
    img_dir = "../dataset/Train/frames/"
    gt_dir = "../dataset/Train/annot/"
    val_dir = "../dataset/Test/frames/"
    val_gt_dir = "../dataset/Test/annot/"
    set_wt_save_path = "./save_model/"
    set_wt_save_name = "3DUnet"
    batchsize = 4
    trainloader = DataLoader(
        Dataset3D(gt_dir, img_dir),
        batch_size = batchsize, shuffle = True, num_workers=4)
    valloader = DataLoader(
        Dataset3D(val_gt_dir, val_dir),
        shuffle = False, num_workers=4
    )

    if not os.path.exists(set_wt_save_path):
        os.makedirs(set_wt_save_path)

    model = unet_2D3D.UNet3D(3, 3)

    weight_decay = 1e-4
    start_epoch = 0
    best_mIoU = 0.0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device type:", device.type)
    if os.path.isfile(set_wt_save_path + set_wt_save_name + '_last.pth'):
        print('**********Resume*************')
        checkpoint = torch.load(set_wt_save_path + set_wt_save_name + '_last.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0006, max_lr=0.0012,
                                          step_size_up=200, cycle_momentum=False)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        # best_mIoU = checkpoint['best_mIoU']
        # best_w_mIoU = checkpoint['best_w_mIoU']
        # best_r_mIoU = checkpoint['best_r_mIoU']
        best_mIoU = 0.0
        best_w_mIoU = 0.0
        best_r_mIoU = 0.0
        print("< save point >")
        print("epoch : ", start_epoch)
        print("Best mIoU : ", best_mIoU)
        print("Best w mIoU : ", best_w_mIoU)
        print("Best r mIoU : ", best_r_mIoU)
    else:
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0006, max_lr=0.0012,
                                                      step_size_up=200, cycle_momentum=False)

    summary = SummaryWriter()
    train_3Dunet(model,optimizer,scheduler,trainloader,valloader,
                 set_wt_save_path,set_wt_save_name,device,best_mIoU,start_epoch,num_epochs = 10000)
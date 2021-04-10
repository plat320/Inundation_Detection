import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import Dataset

import os
from before_ex import vnet_wr

import time
#import matplotlib.pyplot as plt
import cv2


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

def iou_cal(pred, target):
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return intersection / (A_sum + B_sum - intersection)

class Test_Water_Dataset(Dataset):
    def __init__(self, groundtruthdir, videodir, frm_ch_num=16, frm_period =5):
        self.gtdir = groundtruthdir
        self.videodir = videodir
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

        height = 16 * 3 * 5
        width = 16 * 4 * 5

        frms = cv2.imread(frmspath + frmsname[0])# first frame
        frms = cv2.resize(frms, (width, height))
        color = frms.shape[2]
        #region for transforming frms
        frms = np.reshape(frms, (height,width,color,1))

        for num in range(self.f_period, self.f_period*self.f_ch_num, self.f_period):#frame period in video
            frm = cv2.imread(frmspath + frmsname[num])
            frm = cv2.resize(frm, (width, height))
            #region for transforming frm
            frm = np.reshape(frm, (height, width, color, 1))
            frms = np.concatenate((frms, frm), axis=3)

        gt = cv2.imread(self.gtdir + self.gtlist[idx],0)
        gt = cv2.resize(gt,(width, height),interpolation=cv2.INTER_NEAREST)
        # region for transforming gt
        gt = np.reshape(gt, (height, width, 1, 1))

        gt = np.concatenate((gt, gt, gt, gt), axis=3) # [HWC4]
        gt = np.concatenate((gt, gt, gt, gt), axis=3) # [HWC16]

        gt_w = (gt == 255)*1.0 #water groundtruth [0, 1]
        gt_r = (gt == 125)*1.0

        frms = torch.tensor(frms, dtype=torch.float)
        gts_w = torch.tensor(gt_w, dtype=torch.float)
        gts_r = torch.tensor(gt_r, dtype=torch.float)

        frms = frms.permute(2, 3, 0, 1) # C F H W
        gts_w = gts_w.permute(2, 3, 0, 1)
        gts_r = gts_r.permute(2, 3, 0, 1)

        frms = frms / 255

        return frms, gts_w, gts_r

#======================parameter setting===========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_gtdir = '/datahdd/WaterDetection/water_video_dataset/Test/annot/'#"/home/hyeongeun/dataset/Test/annot/"
set_videodir ='/datahdd/WaterDetection/water_video_dataset/Test/frames/' #"/home/hyeongeun/dataset/Test/frames/"
set_model_load_path = '/datahdd/WaterDetection/save_model/vnet/'#"/home/hyeongeun/PycharmProjects/vnet/model_save/"
set_model_name = 'vnet_diceL1_fullsz_bch7_191107_best.pth'#"vnet_Focaldice_period_5_cyLR_batch8_randcropflipnoise_191026_0_best.pth"
set_save_result_img_path = '/datahdd/WaterDetection/save_result/vnet/'#"/home/hyeongeun/PycharmProjects/vnet/result_save/focal_dice/"

set_frm_ch_num = 16
set_frm_period = 5
#==================================================================

print("="*20)
print("Check before Testing")
gtdir = set_gtdir
videodir = set_videodir
print("Ground truth image dir : ", gtdir)
print("Frames dir : ", videodir)

print("-"*10)
print("Create Testset class")
frm_ch_num = set_frm_ch_num
frm_period = set_frm_period
print("# of frames : ", frm_ch_num)
print("frame period : ", frm_period)
testset = Test_Water_Dataset(gtdir,videodir,frm_ch_num,frm_period)
print("Testset length : ",testset.__len__())
frms, gt_w, gt_r = testset[0] #,gt_r
data_height = frms.shape[2]
data_width = frms.shape[3]
print("Frms shape : ",frms.shape)
print("Water ground truth shape : ", gt_w.shape)
print("Road ground truth shape : ", gt_r.shape)

print("-"*10)
print("Create Testset dataloader")
batch_sz = 1 # just 1 becaz folder name
print("Batch size : ",batch_sz)
testloaders =torch.utils.data.DataLoader(testset, batch_size = batch_sz, shuffle= False)
videos, gts_w, gts_r = next(iter(testloaders)) #, gts_r , s_r
print("Videos shape : ",videos.shape)
print("Water gts shape : ",gts_w.shape)
print("Road gts shape : ",gts_r.shape)

print("-"*10)
print("Create model")
model_load_path= set_model_load_path + set_model_name
model= vnet_wr.VNet(elu=True, nll=False, frm_ch=frm_ch_num, height=data_height, width=data_width)

checkpoint = torch.load(model_load_path)
model.load_state_dict(checkpoint['model_state_dict'])
print("< save point >")
print("epoch : ",checkpoint['epoch'])
print("Best epoch loss : ",checkpoint['best_epoch_loss'])
print("-- Water epoch loss : ",checkpoint['water_epoch_loss'])
print("-- Road epoch loss : ",checkpoint['road_epoch_loss'])
#'epoch':epoch, 'model_state_dict':model.state_dict(), 'optimizer_state_dict':optimizer.state_dict(),'best_epoch_loss':best_epoch_loss,'water_epoch_loss':water_epoch_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = False
if device.type == "cuda":
    model.to(device)
    use_gpu = True
print("Device type:", device.type)

model.eval()

print("="*20)
print("Start test")
epoch_start_time =time.time()
cnt = 0
with torch.no_grad():
    Test_w_dice = 0.0
    Test_w_l1 = 0.0
    Test_r_dice = 0.0
    Test_r_l1 = 0.0
    Total_w_mIoU = 0.0
    Total_r_mIoU = 0.0
    Total_max_w_mIoU = 0.0
    Total_max_r_mIoU = 0.0
    for frms, gts_w, gts_r in testloaders:
        cnt += 1
        if use_gpu:
            frms, gts_w, gts_r = frms.cuda(), gts_w.cuda(), gts_r.cuda()
        frms, gts_w, gts_r = Variable(frms), Variable(gts_w), Variable(gts_r)
        output = model(frms)
        pred_water = output[:, :, 0]
        pred_road = output[:, :, 1]

        Test_w_dice += dice_loss(pred_water, gts_w)
        Test_w_l1 += L1_loss(pred_water, gts_w)
        Test_r_dice += dice_loss(pred_road, gts_r)
        Test_r_l1 += L1_loss(pred_road, gts_r)

        w_pred = pred_water.contiguous().view(frm_ch_num, data_height, data_width)
        w_pred_thr = (w_pred > 0.5)*1.0

        r_pred = pred_road.contiguous().view(frm_ch_num, data_height, data_width)
        r_pred_thr = (r_pred > 0.5) * 1.0

        w_mIoU = iou_cal(w_pred_thr, gts_w)
        r_mIoU = iou_cal(r_pred_thr, gts_r)

        Total_w_mIoU += w_mIoU
        Total_r_mIoU += r_mIoU
        print(cnt,' --- w_mIoU :', w_mIoU,' / r_mIoU :', r_mIoU)

        dirpath = set_save_result_img_path + set_model_name[:-4] + "/%05d/" % cnt
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        frms_FHWC = frms[0,:,:,:,:].permute(1,2,3,0)
        max_w_mIoU = 0
        max_r_mIoU = 0
        for i in range(frm_ch_num):
            frm = frms_FHWC[i,:,:,:]*255
            wpred_thr = w_pred_thr[i, :, :]
            rpred_thr = r_pred_thr[i, :, :]
            if max_w_mIoU < iou_cal(wpred_thr, gts_w[:,:,i,:,:]):
                max_w_mIoU = iou_cal(wpred_thr, gts_w[:,:,i,:,:])
            if max_r_mIoU < iou_cal(rpred_thr, gts_r[:,:,i,:,:]):
                max_r_mIoU = iou_cal(rpred_thr, gts_r[:,:,i,:,:])

            over = frms_FHWC[i,:,:,:]*255
            over[:, :, 0] = (1 - wpred_thr) * frm[:, :, 0] + wpred_thr * 255
            over[:, :, 1] = (1 - wpred_thr) * frm[:, :, 1]
            over[:, :, 2] = (1 - wpred_thr) * frm[:, :, 2]

            over[:, :, 0] = (1 - rpred_thr) * over[:, :, 0]
            over[:, :, 1] = (1 - rpred_thr) * over[:, :, 1] + rpred_thr * 255
            over[:, :, 2] = (1 - rpred_thr) * over[:, :, 2]

            wpred_thr = wpred_thr * 255
            rpred_thr = rpred_thr*255

            frm = frm.cpu().detach().numpy()
            wpred_thr = wpred_thr.cpu().detach().numpy()
            rpred_thr = rpred_thr.cpu().detach().numpy()
            over = over.cpu().detach().numpy()

            frm_dirpath = dirpath + "frm_%03d.jpg" % i
            wpred_dirpath = dirpath + "wpred_%03d.jpg" % i
            rpred_dirpath = dirpath + "rpred_%03d.jpg" % i
            over_dirpath = dirpath + "over_%03d.jpg" % i

            cv2.imwrite(frm_dirpath,frm)
            cv2.imwrite(wpred_dirpath,wpred_thr)
            cv2.imwrite(rpred_dirpath, rpred_thr)
            cv2.imwrite(over_dirpath, over)
        print('max_w_mIoU : ',max_w_mIoU, ' / max_r_mIoU : ',max_r_mIoU)
        Total_max_w_mIoU += max_w_mIoU
        Total_max_r_mIoU += max_r_mIoU

    Test_w_dice /= len(testloaders)
    Test_w_l1 /= len(testloaders)
    Test_r_dice /= len(testloaders)
    Test_r_l1 /= len(testloaders)
    Total_w_mIoU /= len(testloaders)
    Total_r_mIoU /= len(testloaders)
    Total_max_w_mIoU /= len(testloaders)
    Total_max_r_mIoU /= len(testloaders)

    print("Test result : Water dice loss = {:.5f}, Water L1 loss = {:.5f}".format(Test_w_dice,Test_w_l1))
    print("Test result : Road dice loss = {:.5f}, Road L1 loss = {:.5f}".format(Test_r_dice, Test_r_l1))
    print("Test result : Total_w_mIoU = {:.5f}, Total_r_mIoU = {:.5f}".format(Total_w_mIoU, Total_r_mIoU))
    print("Test result : Total_max_w_mIoU = {:.5f}, Total_max_r_mIoU = {:.5f}".format(Total_max_w_mIoU, Total_max_r_mIoU))

epoch_time = time.time() - epoch_start_time
#time_h = int((epoch_time // 60) // 60)
time_m = int((epoch_time // 60) % 60)
time_s = epoch_time % 60
print("Time : {:.0f}min {:.1f}sec".format(time_m,time_s))
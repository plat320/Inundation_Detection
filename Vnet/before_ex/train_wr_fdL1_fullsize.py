import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

import os
from before_ex import vnet_wr

import time
#import matplotlib.pyplot as plt
import cv2


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
        #flip_flag = np.random.randint(2)

        height = 16*3*5
        width = 16*4*5

        frms = cv2.imread(frmspath + frmsname[0])# first frame
        frms = cv2.resize(frms, (width, height))
        #show_frms = torch.tensor(frms.copy(), dtype=torch.float)

        color = frms.shape[2]
        frms = np.reshape(frms, (height, width, color, 1))

        for num in range(self.f_period, self.f_period*self.f_ch_num, self.f_period):#frame period in video
            frm = cv2.imread(frmspath + frmsname[num])
            frm = cv2.resize(frm, (width, height))
            #region for transforming frm
            frm = np.reshape(frm, (height, width, color, 1))
            frms = np.concatenate((frms, frm), axis=3)

        gt = cv2.imread(self.gtdir + self.gtlist[idx],0)
        gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)

        gt_w = (gt == 255) * 1.0  # water groundtruth [0, 1]
        gt_r = (gt == 125) * 1.0

        #show_gt_w = torch.tensor(gt_w.copy(), dtype=torch.float)
        #show_gt_r = torch.tensor(gt_r.copy(), dtype=torch.float)

        gt = np.reshape(gt, (height, width, 1, 1))

        gt = np.concatenate((gt, gt, gt, gt), axis=3)  # [HWC4]
        gt = np.concatenate((gt, gt, gt, gt), axis=3)  # [HWC16]

        gt_w = (gt==255)*1.0 #water groundtruth [0, 1]
        gt_r = (gt==125)*1.0

        frms = torch.tensor(frms, dtype=torch.float)
        gts_w = torch.tensor(gt_w, dtype=torch.float)
        gts_r = torch.tensor(gt_r, dtype=torch.float)

        frms = frms.permute(2, 3, 0, 1)  # C F H W
        gts_w = gts_w.permute(2, 3, 0, 1)
        gts_r = gts_r.permute(2, 3, 0, 1)

        frms = frms / 255

        return frms, gts_w, gts_r

'''
class CCTVDataset(Dataset):
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
        flip_flag = np.random.randint(2)

        height = 9 * 32
        width = 16 * 32

        frms = io.imread(frmspath + frmsname[0])# first frame
        frms = cv2.resize(frms, (width, height))

        gt = io.imread(self.gtdir + self.gtlist[idx])
        gt = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST)
        gt = gt[:,:,0:3]

        gt_w = gt==255 #water groundtruth [0, 1]
        gt_r = gt==125

        r_frms = np.reshape(frms[:, :, 0], (height, width, 1, 1))
        g_frms = np.reshape(frms[:, :, 1], (height, width, 1, 1))
        b_frms = np.reshape(frms[:, :, 2], (height, width, 1, 1))
        gts_w = gt_w.copy()
        gts_r = gt_r.copy()


        for num in range(self.f_period, self.f_period*self.f_ch_num, self.f_period):#frame period in video
            frm = io.imread(frmspath + frmsname[num])
            frm = cv2.resize(frm, (width, height))

            r_frm = np.reshape(frm[:, :, 0], (height, width, 1, 1))
            g_frm = np.reshape(frm[:, :, 1], (height, width, 1, 1))
            b_frm = np.reshape(frm[:, :, 2], (height, width, 1, 1))

            r_frms = np.concatenate((r_frms, r_frm), axis=2)
            g_frms = np.concatenate((g_frms, g_frm), axis=2)
            b_frms = np.concatenate((b_frms, b_frm), axis=2)

            gts_w = np.concatenate((gts_w, gt_w), axis=2)
            gts_r = np.concatenate((gts_r, gt_r), axis=2)

        frms = np.concatenate((r_frms, g_frms, b_frms), axis=3)

        gt_w2 = np.reshape(gts_w[:, :, 0:self.f_ch_num], (height, width, self.f_ch_num, 1))
        gt_r2 = np.reshape(gts_r[:, :, 0:self.f_ch_num], (height, width, self.f_ch_num, 1))

        frms = torch.tensor(frms, dtype=torch.float)
        gt_w2 = torch.tensor(gt_w2, dtype=torch.float)
        gt_r2 = torch.tensor(gt_r2, dtype=torch.float)

        frms = frms.permute(3, 2, 0, 1)
        gt_w2 = gt_w2.permute(3, 2, 0, 1)
        gt_r2 = gt_r2.permute(3, 2, 0, 1)

        frms = frms / 255

        return frms, gt_w2, gt_r2
'''

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

def diceL1_loss(pred, target):
    return (dice_loss(pred,target) + L1_loss(pred,target))/2

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

set_frm_ch_num = 16
set_frm_period = 5
set_batch_size = 7
set_base_lr = 0.0006  # scheduler setting
set_max_lr = 0.0012  # scheduler setting
set_step_size_up = 200  # scheduler setting
set_step_size_down = 200

set_wt_save_name = 'vnet_diceL1_fullsz_bch14_191110'
where = "server" #"home" #"lab"
print(where)

if where == "lab":
    set_gtdir = "/datahdd/dataset/water_segmentation/Train/annot/"
    set_videodir = "/datahdd/dataset/water_segmentation/Train/frames/"
    set_wt_save_path = "/datahdd/code/water detection/vnet.pytorch/model_save/"

elif where == "home":
    set_gtdir = "/home/hyeongeun/dataset/Train/annot/"
    set_videodir = "/home/hyeongeun/dataset/Train/frames/"
    set_wt_save_path = "/home/hyeongeun/PycharmProjects/vnet/model_save/"

elif where == "server":
    set_gtdir = '/datahdd/WaterDetection/water_video_dataset/Train/annot/'
    set_videodir = "/datahdd/WaterDetection/water_video_dataset/Train/frames/"
    set_wt_save_path = "/datahdd/WaterDetection/save_model/vnet/"

else :
    raise Exception("Input 'where'.")

#--------------------
writer = SummaryWriter()
print("="*30)
print("Check before training")
gtdir =  set_gtdir
videodir = set_videodir
print("Ground truth image dir : ", gtdir)
print("Frames dir : ", videodir)

print("-"*20)
print("Create dataset class")


frm_ch_num = set_frm_ch_num
frm_period = set_frm_period
print("# of frames : ", frm_ch_num)
print("frame period : ", frm_period)
cctv_dataset = CCTVDataset(gtdir,videodir,frm_ch_num,frm_period)
print("Dataset length : ",cctv_dataset.__len__())
frms, gt_w, gt_r = cctv_dataset[0] #,gt_r
print("Frms shape : ",frms.shape)
print("Water ground truth shape : ", gt_w.shape)
print("Road ground truth shape : ", gt_r.shape)
data_height = frms.shape[2]
data_width = frms.shape[3]

print("-"*20)
print("Create dataloader")
batch_sz = set_batch_size
print("Batch size : ",batch_sz)
dataloaders =torch.utils.data.DataLoader(cctv_dataset, batch_size = batch_sz, shuffle= True, num_workers=8)
videos, gts_w, gts_r  = next(iter(dataloaders)) #, gts_r , s_r
print("Videos shape : ",videos.shape)
print("Water gts shape : ",gts_w.shape)
print("Road gts shape : ",gts_r.shape)
#print("test v w r : ", s_v.shape, s_w.shape, s_r.shape) #, s_r.shape)

print("-"*20)
print("Create model")
model= vnet_wr.VNet(elu=True, nll=False, frm_ch=frm_ch_num, height=data_height, width=data_width)
weight_decay = 1e-4
start_epoch = 0
if os.path.isfile(set_wt_save_path+set_wt_save_name+'_last.pth'):
    print('**********Resume*************')
    checkpoint = torch.load(set_wt_save_path+set_wt_save_name+'_last.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("< save point >")
    print("epoch : ", checkpoint['epoch'])
    print("Best epoch loss : ", checkpoint['best_epoch_loss'])
    print("-- Water epoch loss : ", checkpoint['water_epoch_loss'])
    print("-- Road epoch loss : ", checkpoint['road_epoch_loss'])


#print("Weight decay : ",weight_decay)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Device type:", device.type)

model.train()

optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
scheduler = lr_scheduler.CyclicLR(optimizer, base_lr = set_base_lr, max_lr=set_max_lr,
                                  step_size_up = set_step_size_up, cycle_momentum=False)

if os.path.isfile(set_wt_save_path+set_wt_save_name+'_last.pth'):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epochs = 10000
best_epoch_loss = 1.0
best_epoch_loss_w = 1.0
best_epoch_loss_r = 1.0

best_water_loss = 1.0
best_road_loss = 1.0

print("="*30)
print("Start train")
for epoch in range(start_epoch,epochs):
    epoch_start_time =time.time()
    epoch_loss = 0.0
    water_epoch_loss = 0.0
    road_epoch_loss = 0.0

    show10_video = None
    show10_w_gt = None
    show10_w_pred = None
    show10_r_gt = None
    show10_r_pred = None

    show20_video = None
    show20_w_gt = None
    show20_w_pred = None
    show20_r_gt = None
    show20_r_pred = None

    for batch_idx, (frms, gts_w, gts_r) in enumerate(dataloaders): #, gts_r, show_gts_r
        frms, gts_w, gts_r = frms.to(device), gts_w.to(device),gts_r.to(device)#, gts_r.cuda()#, gts_r
        optimizer.zero_grad()
        output = model(frms)
        pred_water = output[:, :, 0]
        pred_road = output[:, :, 1]

        water_loss = (dice_loss(pred_water,gts_w) + L1_loss(pred_water,gts_w))/2#,batch_sz)
        road_loss = (dice_loss(pred_road,gts_r) + L1_loss(pred_road,gts_r) )/2#,batch_sz)

        loss = (water_loss + road_loss)/2 #L1_weight*(water_L1_loss) #+ road_loss + road_L1_loss)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        water_epoch_loss += water_loss.item()
        road_epoch_loss += road_loss.item()

        '''
        if batch_idx == 10:
            show10_w_pred = pred_water
            show10_w_gt = show_gts_w.permute(0, 3, 1, 2)
            show10_w_gt = show10_w_gt * 255

            show10_r_pred = pred_road
            show10_r_gt = show_gts_r.permute(0, 3, 1, 2)
            show10_r_gt = show10_r_gt * 255

            show10_video = show_frms.permute(0, 3, 1, 2)

            show10_w_pred = show10_w_pred.view(-1,frm_ch_num, 128, 128)
            show10_w_pred = show10_w_pred[:, 0:3, :, :]

            show10_r_pred = show10_r_pred.view(-1, frm_ch_num, 128, 128)
            show10_r_pred = show10_r_pred[:, 0:3, :, :]

        if batch_idx == 20:
            show20_w_pred = pred_water
            show20_w_gt = show_gts_w.permute(0, 3, 1, 2)
            show20_w_gt = show20_w_gt * 255

            show20_r_pred = pred_road
            show20_r_gt = show_gts_r.permute(0, 3, 1, 2)
            show20_r_gt = show20_r_gt * 255

            show20_video = show_frms.permute(0, 3, 1, 2)

            show20_w_pred = show20_w_pred.view(-1, frm_ch_num, 128, 128)
            show20_w_pred = show20_w_pred[:, 0:3, :, :]

            show20_r_pred = show20_r_pred.view(-1, frm_ch_num, 128, 128)
            show20_r_pred = show20_r_pred[:, 0:3, :, :]
        '''

    epoch_loss /= len(dataloaders)
    water_epoch_loss /= len(dataloaders)
    road_epoch_loss /= len(dataloaders)

    scheduler.step()

    if best_epoch_loss>epoch_loss:
        best_epoch_loss = epoch_loss
        best_epoch_loss_w = water_epoch_loss
        best_epoch_loss_r = road_epoch_loss

        wt_save_path = set_wt_save_path
        wt_save_name = set_wt_save_name + '_best.pth'
        torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_epoch_loss':best_epoch_loss,
            'water_epoch_loss':best_epoch_loss_w,
            'road_epoch_loss':best_epoch_loss_r
        }, wt_save_path+wt_save_name)

    wt_save_path = set_wt_save_path
    wt_save_name = set_wt_save_name +'_last.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict' : scheduler.state_dict(),
        'best_epoch_loss': best_epoch_loss,
        'water_epoch_loss': water_epoch_loss,
        'road_epoch_loss': road_epoch_loss
    }, wt_save_path + wt_save_name)


    epoch_time = time.time() - epoch_start_time
    time_h = int((epoch_time // 60) // 60)
    time_m = int((epoch_time // 60) % 60)
    time_s = epoch_time % 60
    #print(time_h, "h ", time_m, "m ", time_s, "s")

    print("Epoch {}/{} Total loss : {:.8f} ( Best total loss : {:.8f} = focal dice+l1_w {:.8f} + focal dice+l1_r {:.8f} , lr = {} )".format(epoch, epochs - 1, epoch_loss, best_epoch_loss,
                                                                                                                        best_epoch_loss_w, best_epoch_loss_r,
                                                                                                                        scheduler.get_lr()))
    print("Water loss : {:.8f}  /  Road loss : {:.8f} / Time : {:.0f}min {:.1f}sec".format(water_epoch_loss, road_epoch_loss,
                                                                                           time_m,time_s))


    writer.add_scalar('total loss/train', epoch_loss, epoch)
    writer.add_scalar('water focal dice+L1 loss/train', water_epoch_loss, epoch)
    writer.add_scalar('road focal dice+L1 loss/train', road_epoch_loss, epoch)

    '''
    grid_10video = utils.make_grid(show10_video,nrow=4,normalize=True)

    grid_10_w_gt = utils.make_grid(show10_w_gt, nrow=4,normalize=True)
    grid_10_w_pred = utils.make_grid(show10_w_pred, nrow=4,normalize=True)
    grid_10_w_pred_thres = grid_10_w_pred > 0.5

    grid_10_r_gt = utils.make_grid(show10_r_gt, nrow=4, normalize=True)
    grid_10_r_pred = utils.make_grid(show10_r_pred, nrow=4, normalize=True)
    grid_10_r_pred_thres = grid_10_r_pred > 0.5

    grid_20video = utils.make_grid(show20_video,nrow=4,normalize=True)

    grid_20_w_gt = utils.make_grid(show20_w_gt,nrow=4,normalize=True)
    grid_20_w_pred = utils.make_grid(show20_w_pred,nrow=4,normalize=True)
    grid_20_w_pred_thres = grid_20_w_pred > 0.5

    grid_20_r_gt = utils.make_grid(show20_r_gt, nrow=4, normalize=True)
    grid_20_r_pred = utils.make_grid(show20_r_pred, nrow=4, normalize=True)
    grid_20_r_pred_thres = grid_20_r_pred > 0.5

    writer.add_image('grid_10_video',grid_10video, epoch)
    writer.add_image('grid_10_water_gt', grid_10_w_gt, epoch)
    writer.add_image('grid_10_water_pred', grid_10_w_pred, epoch)
    writer.add_image('grid_10_water_pred_thres', grid_10_w_pred_thres, epoch)
    writer.add_image('grid_10_road_gt', grid_10_r_gt, epoch)
    writer.add_image('grid_10_road_pred', grid_10_r_pred, epoch)
    writer.add_image('grid_10_road_pred_thres', grid_10_r_pred_thres, epoch)

    writer.add_image('grid_20_video', grid_20video, epoch)
    writer.add_image('grid_20_water_gt', grid_20_w_gt, epoch)
    writer.add_image('grid_20_water_pred', grid_20_w_pred, epoch)
    writer.add_image('grid_20_water_pred_thres', grid_20_w_pred_thres, epoch)
    writer.add_image('grid_20_road_gt', grid_20_r_gt, epoch)
    writer.add_image('grid_20_road_pred', grid_20_r_pred, epoch)
    writer.add_image('grid_20_road_pred_thres', grid_20_r_pred_thres, epoch)
    '''



    ##----------------------------------------------end of each epoch------------------------------------------------------------------





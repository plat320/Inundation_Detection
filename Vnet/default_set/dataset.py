import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
from torchvision import transforms

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
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
import sys
import torch.optim as optim
import torch.nn.functional as F
import cv2
import test

sys.path.append(os.path.abspath(".") + "/segmentation_models_pytorch/unet")
import model


def iou_cal(pred, target):
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    if (A_sum+B_sum-intersection) <= 0:
        return 0
    else:
        return intersection / (A_sum + B_sum - intersection)


def get_IoU(gt, pred):
    gt = gt.flatten()
    size = gt.size
    pred = pred.flatten()
    TP = np.multiply(gt, pred)
    TP = TP.astype('uint8')
    TP = np.sum(TP)
    n_gt = np.logical_not(gt)
    n_gt = n_gt.astype('uint8')
    n_pred = np.logical_not(pred)
    n_pred = n_pred.astype('uint8')
    TN = np.multiply(n_gt, n_pred)
    TN = np.sum(TN)
    if TN == size:
        IoU = 100
    else:
        IoU = 100*((TP*1.0)/(size-TN))
    return IoU

class water_fcn_dataset(Dataset):
    def __init__(self, imagedir, groundtruthdir):
        self.gtdir = groundtruthdir
        self.imgdir = imagedir
        self.imglist = sorted(os.listdir(imagedir))
        self.gtlist = sorted(os.listdir(groundtruthdir))

    def __len__(self):
        return len(self.imglist)


    def __getitem__(self, idx):
        img = Image.open(self.imgdir + self.imglist[idx])
        gt = cv2.imread(self.gtdir + self.gtlist[idx])
        width, height = img.size
        if width > 1000 or height >1000:
            width = round(width/2)
            height = round(height/2)
        img = img.resize((width, height))
        img = np.array(img)
        gt = cv2.resize(gt,dsize=(width,height), interpolation = cv2.INTER_NEAREST)
        gt = np.asarray(gt)

        wgt = gt == 255
        rgt = gt == 125
        wgt = wgt * 1.0
        rgt = rgt * 1.0
        rgt = rgt[:,:,:1]
        wgt = wgt[:,:,:1]

        img = Image.fromarray(img)

        wgt = np.squeeze(wgt, axis = 2)
        rgt = np.squeeze(rgt, axis = 2)

        wgt = torch.from_numpy(wgt)
        rgt = torch.from_numpy(rgt)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(img)
        wgt = wgt.type(torch.FloatTensor)
        rgt = rgt.type(torch.FloatTensor)

        return img, wgt, rgt


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

if __name__ == '__main__':
# def test():
    img_dir = "../dataset/Test/image/"
    gt_dir = "../dataset/Test/annot/"
    batchsize = 4
    set_base_lr = 0.0004  # scheduler setting
    set_max_lr = 0.0012  # scheduler setting
    set_step_size_up = 50  # scheduler setting
    weight_decay = 1e-4

    testloader = DataLoader(
        water_fcn_dataset(img_dir, gt_dir),
        shuffle = False, num_workers=4)

    model = model.Unet()
    model.load_state_dict(torch.load("./mIoU_2DUnet68.41212198752143.pth"))
    model.eval()
    stime = time.time()
    road_IoU = 0.0
    water_IoU = 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, wlabels, rlabels = data
            org_img = (inputs.squeeze(0).numpy()+3)*40
            model.zero_grad()
            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                model.to('cuda')
                wlabels = wlabels.to('cuda')
                rlabels = rlabels.to('cuda')
            outputs = model(inputs)
            outputs = outputs.squeeze(0).argmax(0)
            water_mask = outputs == 1
            road_mask = outputs == 2
            road_IoU += iou_cal(road_mask, rlabels)
            water_IoU += iou_cal(water_mask, wlabels)
            result_img = np.zeros((3,320,576))
            result_img[1, road_mask.cpu().numpy()] = 255
            result_img[2, water_mask.cpu().numpy()] = 255
            result_img = result_img*0.3 + org_img*0.7
            result_img = Image.fromarray(result_img.astype(dtype=np.uint8).transpose(1,2,0))
            result_img.save("./2Dresult/%d.png"%i)


        road_IoU /= len(testloader)
        water_IoU /= len(testloader)
        print("water_mIoU = %.5f, road_mIoU = %.5f"%(water_IoU, road_IoU))
    # return road_IoU, water_IoU

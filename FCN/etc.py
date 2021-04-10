import urllib
import torch.nn as nn
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import torch.optim as optim
import torch.nn.functional as F
from distutils.version import LooseVersion
import matplotlib.pyplot as plt
import cv2 as cv

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
        # gt = Image.open(self.gtdir + self.gtlist[idx])
        # img = cv.imread(self.imgdir + self.imglist[idx])
        gt = cv.imread(self.gtdir + self.gtlist[idx])
        width, height = img.size
        if width > 1000 or height >1000:
            width = round(width/2)
            height = round(height/2)
        img = img.resize((width, height))
        ################수정
        img = np.array(img)
        ################수정
        gt = np.asarray(gt)

        gt1 = gt == 255
        gt2 = gt == 125
        gt1 = gt1 * 1.0
        gt2 = gt2 * 2.0
        gt = gt1 + gt2
        gt = cv.resize(gt,dsize=(width,height), interpolation = cv.INTER_NEAREST)
        gt = gt[:,:,:1]

        #################수정

        # # 여기가 randomcrop
        # img, rand_x, rand_y = RandomCrop(img, 128)
        # gt, _, _ = RandomCrop(gt, 128, rand_x, rand_y)
        #
        # # 여기가 randomflip
        # img, flip = RandomFlip(img)
        # gt, _ = RandomFlip(gt, flip)
        #
        # img = Image.fromarray(img)
        # plt.imshow(img)
        # ################수정

        gt = np.squeeze(gt, axis = 2)
        gt1 = Image.fromarray(gt)
        plt.imshow(gt)
        gt = torch.from_numpy(gt)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # preprocess1 = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        #
        # gt = preprocess1(gt)
        img = preprocess(img).cpu()
        gt = gt.type(torch.LongTensor)

        return img, gt

def RandomCrop(img, size, x = 0, y = 0):       # img format is numpy w x h x c
    if x == 0 and y ==0:
        width, height = img.shape[:-1]
        x = random.randrange(width - size)
        y = random.randrange(height - size)
    img = img[x : x+size, y : y+size, :]

    return img, x, y

def RandomFlip(img, flip = -1):       # img format is numpy w x h x c
    if flip == -1:
        flip = random.randrange(4)

    if flip == 0:
        img = img
    elif flip == 1:
        img = np.fliplr(img).copy()
    elif flip == 2:
        img = np.flipud(img).copy()
    elif flip == 3:
        img = np.fliplr(img).copy()
        img = np.flipud(img).copy()
    img = img

    return img, flip


if __name__ == '__main__':



    img_dir = "./water/train/image/"
    gt_dir = "./water/train/annot/"
    trainloader = DataLoader(
        water_fcn_dataset(img_dir, gt_dir),
        batch_size = 1, shuffle = True)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        img = (inputs.cpu().numpy()+3)*40
        img = img.astype(dtype = np.uint8).squeeze(0)
        img = Image.fromarray(img.transpose(1,2,0))
        img.show()
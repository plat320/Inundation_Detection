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
import cv2 as cv
import test
import sys

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
        gt = cv.resize(gt,dsize=(width,height), interpolation = cv.INTER_NEAREST)
        gt = np.asarray(gt)

        wgt = gt == 255
        rgt = gt == 125
        wgt = wgt * 1.0
        rgt = rgt * 1.0
        # gt = gt1 + gt2
        rgt = rgt[:,:,:1]
        wgt = wgt[:,:,:1]

        #################수정

        # # 여기가 randomcrop
        # img, rand_x, rand_y = RandomCrop(img, 128)
        # gt, _, _ = RandomCrop(gt, 128, rand_x, rand_y)
        #
        # # 여기가 randomflip
        # img, flip = RandomFlip(img)
        # gt, _ = RandomFlip(gt, flip)

        img = Image.fromarray(img)

        ################수정

        wgt = np.squeeze(wgt, axis = 2)
        rgt = np.squeeze(rgt, axis = 2)

        wgt = torch.from_numpy(wgt)
        rgt = torch.from_numpy(rgt)
        # gt = Image.fromarray(gt)

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # preprocess1 = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        #
        # gt = preprocess1(gt)
        img = preprocess(img)
        # wgt = wgt/255.0
        # rgt = rgt/255.0
        wgt = wgt.type(torch.FloatTensor)
        rgt = rgt.type(torch.FloatTensor)
        # gt = preprocess(gt)
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
    img_dir = "../dataset/Train/image/"
    gt_dir = "../dataset/Train/annot/"
    set_base_lr = 0.0004  # scheduler setting                # 기존 set_base_lr = 0.0004
    set_max_lr = 0.0012  # scheduler setting                # 기존 set_max_lr = 0.0012
    set_step_size_up = 100  # scheduler setting              # 기존 500
    weight_decay = 1e-4
    batchsize = 16
    trainloader = DataLoader(
        water_fcn_dataset(img_dir, gt_dir),
        batch_size = batchsize, shuffle = True, num_workers=2)

    summary = SummaryWriter()


    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    for params in model.parameters():
        params.requires_grad = False
    model.aux_classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
    model.classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
    model.classifier._modules['0'].convs._modules['4']._modules['0'] = nn.AdaptiveAvgPool2d(output_size = 16)
    model.aux_classifier.add_module('5', nn.Softmax(dim = 1))
    model.classifier.add_module('5', nn.Softmax(dim = 1))
    model.train()

    # optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=set_base_lr, max_lr=set_max_lr,
    #                                               step_size_up = set_step_size_up, cycle_momentum=False)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # criterion = nn.CrossEntropyLoss()
    prev_loss = 1000000.0
    stime = time.time()
    prev_mIoU = 0.0

    # model.load_state_dict(torch.load("./epoch_DeepLabv3.pth"))

    for epoch in range(10000):
        running_loss = 0.0
        twloss = 0
        trloss = 0
        tlwloss = 0
        tlrloss = 0

        for i, data in enumerate(trainloader, 0):
            inputs, wlabels, rlabels = data

            model.zero_grad()

            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                model.to('cuda')
                wlabels = wlabels.to('cuda')
                rlabels = rlabels.to('cuda')

            outputs = model(inputs)['out']
            # outputs = outputs.unsqueeze(0)
            woutputs = outputs[:,1,:,:]
            routputs = outputs[:,2,:,:]
            # nanoutputs = outputs[0,:,:]
            wloss = dice_loss(woutputs, wlabels)
            lwloss = L1_loss(woutputs, wlabels)
            rloss = dice_loss(routputs, rlabels)
            lrloss = L1_loss(routputs, rlabels)
            loss = wloss + rloss + lwloss + lrloss
            # a = list(model.parameters())[0].clone()
            loss.backward()
            # print(model.classifier.grad)
            optimizer.step()
            # b = list(model.parameters())[0].clone()
            # print(torch.equal(a.data, b.data))
            running_loss += loss.item()
            twloss += wloss
            trloss += rloss
            tlwloss += lwloss
            tlrloss += lrloss

            # if i%20 == 19:
                # print('[%d, %5d] loss: %.5f lr: %f' %
                      # (epoch, i + 1, running_loss / (10 * batchsize), scheduler.get_lr()[0]))
        print('[%d, %5d] loss: %.5f' %(epoch, i + 1, running_loss / len(trainloader)))
        if prev_loss > running_loss:
            PATH = './loss_DeepLabv3.pth'
            print("save lowest loss path")
            torch.save(model.state_dict(), PATH)
            prev_loss =running_loss                     # epoch마다 저장하는거 따로
        running_loss = 0.0

        # scheduler.step()

        output_predictions = outputs[0].argmax(0)
        pred = output_predictions.cpu().numpy()
        width, height = pred.shape[0:]
        wpred = pred == 1
        rpred = pred == 2
        wpred = wpred * 255
        rpred = rpred * 255
        output_numpy = np.zeros((3, width, height), dtype=np.uint8)
        output_numpy[1, :, :] = wpred.astype(dtype=np.uint8)
        output_numpy[2, :, :] = rpred.astype(dtype=np.uint8)
        summary.add_image('image/output_img', output_numpy, epoch)


        img = (inputs.cpu().numpy()[0]+3) * 40
        img = img.astype(dtype = np.uint8)
        summary.add_image('image/input', img, epoch)        # input images look pretty good

        summary.add_scalar('loss/dice_water_loss', twloss/int(484 / batchsize), epoch)
        summary.add_scalar('loss/dice_road_loss', trloss/int(484 / batchsize), epoch)
        summary.add_scalar('loss/l1_water_loss', tlwloss/int(484 / batchsize), epoch)
        summary.add_scalar('loss/l1_road_loss', tlrloss/int(484 / batchsize), epoch)
        summary.add_scalar('loss/loss', (twloss+trloss+tlwloss+tlrloss)/(484*4 / batchsize), epoch)


        print('epoch = %d, execute time = %.3f' %(epoch, time.time() - stime))
        stime = time.time()
        PATH = './epoch_DeepLabv3.pth'
        torch.save(model.state_dict(), PATH)

        # state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
        #          'optimizer': optimizer.state_dict(),  'prev_loss' : prev_loss}
        # torch.save(state, "checkpoint")

        if epoch %10 ==0:
            now_mIoU, rmIoU = test.test()
            summary.add_scalar('mIoU/water_mIoU', now_mIoU, epoch)
            summary.add_scalar('mIoU/road_mIoU', rmIoU, epoch)

            torch.save(model.state_dict(), "./mIoU_DeepLabv3" + str(now_mIoU) + ".pth")
            print("Best mIoU path")

    print('Finished Training')


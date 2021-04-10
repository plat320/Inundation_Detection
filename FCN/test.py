import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class test_data(Dataset):
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
        img = np.array(img)
        gt = np.asarray(gt)

        gt1 = gt == 255
        gt2 = gt == 125
        gt1 = gt1 * 1.0
        gt2 = gt2 * 2.0
        gt = gt1 + gt2
        gt = cv.resize(gt,dsize=(width,height), interpolation = cv.INTER_NEAREST)
        gt = gt[:,:,:1]
        gt = np.squeeze(gt, axis = 2)
        gt = torch.from_numpy(gt)
        img = torch.from_numpy(img)
        gt = gt.type(torch.cuda.LongTensor)

        return img, gt


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


# if __name__ == '__main__':
def test():
    img_dir = "../dataset/Test/image/"
    gt_dir = "../dataset/Test/annot/"

    testloader = DataLoader(
        test_data(img_dir, gt_dir),
        batch_size=1, shuffle=False)

    model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
    model.aux_classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
    model.classifier._modules['4'] = nn.Conv2d(512,3,kernel_size=(1,1),stride=(1,1))
    model.aux_classifier.add_module('5', nn.Softmax(dim = 1))
    model.classifier.add_module('5', nn.Softmax(dim = 1))
    model.eval()
    model.load_state_dict(torch.load("./epoch_FCN.pth"))
                                                                                # IoU GT 펼치고 이미지 펼치고 곱해서 output 0.5이상 thresholding해서 뽑아내야함
    total = 0
    correct = 0
    i=0
    RoadmIoU = 0
    WatermIoU = 0
    result_path = "./result/"
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.squeeze(0)
            org = Image.fromarray(images.byte().cpu().numpy())
            # org.save(result_path + "image_%d.png" %i)

            images = Image.fromarray(images.byte().cpu().numpy())
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            images = preprocess(images)
            images = images.unsqueeze(0)
            if torch.cuda.is_available():
                images = images.to('cuda')
                model.to('cuda')
                labels = labels.to('cuda')
            outputs = model(images)['out'][0]
            output_predictions = outputs.argmax(0)
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for  i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
            pred = output_predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            water = labels == 1
            water = water * 1.0
            road = labels == 2
            road = road * 1.0

            p_water = pred == 1
            p_water = p_water * 1.0
            p_road = pred == 2
            p_road = p_road * 1.0

            # width, height = labels.shape[1:]
            # water = np.zeros(1,width,height)
            # road = np.zeros(1,width,height)
            # for i in range(width):
            #     for j in range(height):
            #         if labels[0,i,j] == 1:
            #             water[0,i,j] =1
            #         if labels[]


            RoadIoU = get_IoU(road, p_road)
            WaterIoU = get_IoU(water,p_water)
            RoadmIoU = RoadmIoU + RoadIoU
            WatermIoU = WatermIoU + WaterIoU
            predict = Image.fromarray(output_predictions.byte().cpu().numpy())
            predict.putpalette(colors)
            # plt.imsave(result_path + "image_%d_0.png" %i, predict)
            i += 1
    RoadmIoU = RoadmIoU / i
    WatermIoU = WatermIoU / i
    print("RoadmIoU =  %f" %RoadmIoU)
    print("WatermIoU =  %f" %WatermIoU)

    return WatermIoU, RoadmIoU
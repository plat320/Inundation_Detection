import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    img_dir = "../dataset/Test/image/"
    gt_dir = "../dataset/Test/annot/"

    testloader = DataLoader(
        test_data(img_dir, gt_dir),
        batch_size=1, shuffle=False)

    model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
    model.aux_classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
    model.classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
    model.classifier._modules['0'].convs._modules['4']._modules['0'] = nn.AdaptiveAvgPool2d(output_size = 16)
    model.eval()
    model.load_state_dict(torch.load("./epoch_DeepLabv3.pth"))
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
            org.save(result_path + "image_%d.png" %i)

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
                # labels = labels.to('cuda')
            outputs = model(images)['out'][0]
            output_predictions = outputs.argmax(0)
            pred = output_predictions.cpu().numpy()
            labels = labels.cpu().numpy()

            water = labels == 1
            water = water * 1.0
            road = labels == 2
            road = road * 1.0

            width, height = labels.shape[1:]
            result = np.zeros((3, width, height))
            p_water = pred == 1
            p_water = p_water * 1.0
            result[2] = p_water * 255
            p_road = pred == 2
            p_road = p_road * 1.0
            result[1] = p_road * 255

            result = result.astype(dtype=np.uint8)
            result = Image.fromarray(result.transpose(1,2,0))
            result.save(result_path + 'image_%d_0.png'%i)

            RoadIoU = get_IoU(road, p_road)
            WaterIoU = get_IoU(water, p_water)
            RoadmIoU = RoadmIoU + RoadIoU
            WatermIoU = WatermIoU + WaterIoU
            i += 1
    RoadmIoU = RoadmIoU / i
    WatermIoU = WatermIoU / i
    print("RoadmIoU =  %f" % RoadmIoU)
    print("WatermIoU =  %f" % WatermIoU)

    # return WatermIoU, RoadmIoU
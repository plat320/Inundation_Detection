import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2
import sys
import os
sys.path.append(os.path.abspath(".") + "/code/before_ex")
import vnet_wr

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


def get_road_inundation(water, road):
    full_road = np.sum(road, dtype = 'float32')
    if full_road == 0:
        return -1
    else:
        road = road.flatten()
        water = water.flatten()
        inundation_region = road * water
        inundation_percent = np.sum(inundation_region)/full_road
        return inundation_percent*100

def get_img(num, vid, img, vid_name):
    weight = 0.7
    b = 1-weight
    vid = vid.argmax(axis = 3)
    vid = vid[num,:,:]
    none_mask = vid == 2
    water_mask = vid == 0
    road_mask = vid == 1
    result = np.zeros((3,320,576))
    result[2, water_mask] = 255
    result[1, road_mask] = 255
    result = result * b + img * weight

    cv2.imwrite("./result/"+vid_name+str(num)+".png",result.transpose(1,2,0))

    return water_mask, road_mask


def create_img(img_name, model_name):
    if model_name == "DeepLabV3" or model_name == "FCN":
        if model_name == "DeepLabV3":
            model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
            model.aux_classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
            model.classifier._modules['4'] = nn.Conv2d(256,3,kernel_size=(1,1),stride=(1,1))
            model.classifier._modules['0'].convs._modules['4']._modules['0'] = nn.AdaptiveAvgPool2d(output_size = 16)
            model.eval()
            model.load_state_dict(torch.load("./model/DeepLabv3.pth"))

        elif model_name == "FCN":
            model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
            model.aux_classifier._modules['4'] = nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
            model.classifier._modules['4'] = nn.Conv2d(512, 3, kernel_size=(1, 1), stride=(1, 1))
            model.aux_classifier.add_module('5', nn.Softmax(dim=1))
            model.classifier.add_module('5', nn.Softmax(dim=1))
            model.eval()
            model.load_state_dict(torch.load("./model/FCN.pth"))

        weight = 0.7
        b = 1.0 - weight

        img = Image.open("./flood_img/"+img_name+".jpg")
        org_img = np.array(img)
        ref_road = Image.open("./ref/"+img_name+".jpg")
        tmp_img = img
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = preprocess(img)
        ref_road = preprocess(ref_road)
        img = img.unsqueeze(0)
        ref_road = ref_road.unsqueeze(0)
        gt = np.asarray(cv2.imread("./gt/"+img_name+".png"))#
        height, width = gt.shape[:-1]
        gt1 = gt == 255
        gt2 = gt == 125
        gt1 = np.squeeze(gt1[:,:,:1], axis = 2)
        gt2 = np.squeeze(gt2[:,:,:1], axis = 2)


        gt1 = gt1*1.0
        gt2 = gt2*2.0
        gt = gt1 + gt2
        gt = torch.from_numpy(gt)
        gt = gt.type(torch.cuda.LongTensor)


        with torch.no_grad():
            if torch.cuda.is_available():
                ref_road = ref_road.to('cuda')
                img = img.to('cuda')
                model.to('cuda')
                gt = gt.to('cuda')
            outputs = model(img)['out'][0]
            ref_outputs = model(ref_road)['out'][0]
            pred = outputs.argmax(0).cpu().numpy()
            ref_pred = ref_outputs.argmax(0).cpu().numpy()                # cuda data type에서도 아래 행위들이 되는지..? cpu만 잡아먹고 쓸데없는거같아서 ..
            gt = gt.cpu().numpy()
            ref_pred = ref_pred == 2
            ref_pred = ref_pred * 1.0                   # reference road detect
            water = gt == 1
            water = water * 1.0                         # gt water
            road = gt == 2
            road = road * 1.0                           # gt road
            result = np.zeros((3, height, width))
            p_water = pred == 1
            p_water = p_water * 1.0                     # pred water detect

            inundation_rate = get_road_inundation(p_water, ref_pred)      # inundation detection

            p_road = pred == 2
            p_road = p_road * 1.0                       # pred road detect
            onroad_mask = (p_water[:,:] == 1) & (ref_pred[:,:] == 1)
            water_mask = (p_water[:,:] == 1) & (ref_pred[:,:] == 0)
            road_mask = p_road[:,:] == 1

            org_img = org_img.transpose(2,0,1)
            result[0,onroad_mask] = 255
            result[2,water_mask] = 255
            result[1,road_mask] = 255
            result = result*b + org_img*weight

            result = result.astype(dtype=np.uint8)
            result = Image.fromarray(result.transpose(1,2,0))
            result.save("./result/"+img_name+".png")
            RoadIoU = get_IoU(road, p_road)
            WaterIoU = get_IoU(water, p_water)
        return WaterIoU, RoadIoU, inundation_rate

    elif model_name == "Vnet":
        model = "model"


def get_input(gt_dir,vid_dir, vid_name, height, width, frm_ch_num = 16, frm_period = 5):
    frm_path = vid_dir + "/flood_frames/"+vid_name +"/"
    frm_name = sorted(os.listdir(frm_path))     # 이래야 이름 순서대로나옴
    frms = cv2.imread(frm_path + frm_name[0])           # first frame
    color = frms.shape[2]
    frms = np.reshape(frms, (height, width, color, 1))

    for num in range(frm_period, frm_period * frm_ch_num, frm_period):
        frm = cv2.imread(frm_path + frm_name[num])
        frm = np.reshape(frm, (height, width, color, 1))
        frms = np.concatenate((frms, frm), axis = 3)

    gt = cv2.imread(gt_dir + "/" + vid_name + ".png", cv2.IMREAD_GRAYSCALE)
    gt = np.reshape(gt, (height, width, 1,1))
    gt = np.concatenate((gt,gt,gt,gt), axis = 3)
    gt = np.concatenate((gt,gt,gt,gt), axis = 3)

    gt_w = (gt == 255) * 1.0
    gt_r = (gt == 125) * 1.0
    frms = torch.tensor(frms, dtype=torch.float)
    gt_w = torch.tensor(gt_w, dtype=torch.float)
    gt_r = torch.tensor(gt_r, dtype=torch.float)

    frms = frms.permute(2, 3, 0, 1)  # H W C F => C F H W
    gt_w = gt_w.permute(2, 3, 0, 1)
    gt_r = gt_r.permute(2, 3, 0, 1)

    frms = frms/255
    frms = frms.unsqueeze(0)

    return frms, gt_w, gt_r


if __name__ == '__main__':
    gt_dir = "./gt"
    vid_dir = "./flood_vid"
    vid_name = "cctv"
    frm_ch_num = 16
    frm_period = 5
    data_height = 320
    data_width = 576
    model = vnet_wr.VNet(elu=True, nll=False, frm_ch=frm_ch_num, height=data_height, width=data_width)
    model.eval()

    model.load_state_dict(torch.load("./model/Vnet_fullsize.pth")['model_state_dict'])
    inputs, gt_w, gt_r = get_input(gt_dir, vid_dir,vid_name, data_height, data_width, frm_ch_num, frm_period)

    with torch.no_grad():
        if torch.cuda.is_available():
            # gt_r.to('cuda')
            # gt_w.to('cuda')
            inputs = inputs.to('cuda')
            model.to('cuda')
        outputs = model(inputs)
        outputs = outputs.cpu().numpy()
        outputs = outputs.reshape(frm_ch_num, data_height,data_width, 3)  # N F H W C
        inputs = inputs.cpu().numpy().squeeze(0)* 255
        inputs = inputs.astype(dtype = np.uint8)
    wIoU = 0
    rIoU = 0
    for num in range(frm_ch_num):
        input = inputs[:,num,:,:]
        w_pred, r_pred = get_img(num, outputs,input, vid_name)
        w_pred = w_pred * 1.0
        r_pred = r_pred * 1.0
        wIoU = wIoU + get_IoU(w_pred, gt_w[:,0,:,:].numpy())
        rIoU = rIoU + get_IoU(r_pred, gt_r[:,0,:,:].numpy())
    wIoU = wIoU/frm_ch_num
    rIoU = rIoU/frm_ch_num

    print("wIoU = %.5f rIoU = %.5f" %(wIoU, rIoU))
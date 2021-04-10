import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(".") + "/vnet_code/before_ex")
sys.path.append(os.path.abspath(".") + "/vnet_code/default_set")
sys.path.append(os.path.abspath(".") + "/segmentation_models_pytorch/unet")
import model1
import vnet_wr
import unet_2D3D



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


def get_road_inundation(water, ref_road, ref_water):
    full_road = np.sum(ref_road, dtype = 'float32')
    full_water = np.sum(ref_water, dtype = 'float32')
    full_pred_water = np.sum(water, dtype = 'float32')

    if full_water == 0:
        rate = 0.5
        water_inc_rate = 2
    elif full_road == 0:
        rate = 1.5
        water_inc_rate = full_pred_water / full_water
        if water_inc_rate >= 2:
            water_inc_rate = 2

    else:
        rate = full_water / full_road
        water_inc_rate = full_pred_water/ full_water
        if water_inc_rate >= 2:
            water_inc_rate = 2


    if rate <0.5:
        rate = 0.5
    elif rate > 1.5:
        rate = 1.5

    if full_road == 0:
        return -1
    else:
        ref_road = ref_road.flatten()
        water = water.flatten()
        inundation_region = ref_road * water
        inundation_percent = np.sum(inundation_region) * water_inc_rate / full_road
        # inundation_percent = np.sum(inundation_region) / full_road
        return inundation_percent*100, rate

def get_input(gt_dir,vid_dir, vid_name, height, width, frm_ch_num = 16, frm_period = 5):
    frm_path = vid_dir + "/flood_frames/" + vid_name +"/"
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


def get_img(num, vid, ref_img, img, vid_name, ref_road_mask, ref_water_mask):
    weight = 0.7
    b = 1-weight
    vid = vid.argmax(axis = 3)
    vid = vid[num,:,:]
    water_mask = vid == 0
    road_mask = vid == 1
    on_road_mask =  ref_road_mask * water_mask

    ref_result = np.zeros((3,320,576))
    ref_result[2,ref_water_mask] = 255
    ref_result[1,ref_road_mask] = 255
    ref_result = ref_result*b+ref_img*weight
    ref_result = ref_result.astype(dtype = np.uint8)
    ref_result = Image.fromarray(ref_result.transpose(1,2,0))
    ref_result.save("./result/" + "ref_" + vid_name + str(num).zfill(2) + ".png")


    result = np.zeros((3,320,576))
    only_water_mask = (water_mask[:,:] == 1) & (on_road_mask[:,:] == 0)
    result[2, only_water_mask] = 255
    result[1, road_mask] = 255
    result[0, on_road_mask] = 255
    result = result * b + img * weight
    result = result.astype(dtype = np.uint8)
    result = Image.fromarray(result.transpose(1,2,0))
    result.save("./result/"+vid_name+str(num).zfill(2)+".png")

    return only_water_mask, road_mask, water_mask


def create_img(img_name, model_name):
    ref_rate = 35
    if model_name == "DeepLabV3" or model_name == "FCN" or model_name == "2DUnet":
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

        elif model_name == "2DUnet":
            model = model1.Unet()
            model.load_state_dict(torch.load("./model/2DUnet.pth"))
            model.eval()


        weight = 0.7
        b = 1.0 - weight

        img = Image.open("./flood/"+img_name+".jpg")
        org_img = np.array(img)
        ref_road = Image.open("./ref/"+img_name+".jpg")
        tmp_img = img
        ref_org_img = np.array(ref_road)
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


        with torch.no_grad():
            if torch.cuda.is_available():
                ref_road = ref_road.to('cuda')
                img = img.to('cuda')
                model.to('cuda')
            if model_name == "DeepLabV3" or model_name == "FCN":
                outputs = model(img)['out'][0]
                ref_outputs = model(ref_road)['out'][0]
            else:
                outputs = model(img)[0]
                ref_outputs = model(ref_road)[0]
            pred = outputs.argmax(0).cpu().numpy()
            ref_pred = ref_outputs.argmax(0).cpu().numpy()                # cuda data type에서도 아래 행위들이 되는지..? cpu만 잡아먹고 쓸데없는거같아서 ..
            road_ref_pred = ref_pred == 2
            road_ref_pred = road_ref_pred * 1.0                   # reference road detect
            water_ref_pred = ref_pred == 1
            water_ref_pred = water_ref_pred * 1.0
            water = gt == 1
            water = water * 1.0                         # gt water
            road = gt == 2
            road = road * 1.0                           # gt road
            result = np.zeros((3, height, width))
            ref_result = np.zeros((3, height, width))
            p_water = pred == 1
            p_water = p_water * 1.0                     # pred water detect

            inundation_rate, rate = get_road_inundation(p_water, road_ref_pred, water_ref_pred)      # inundation detection
            base_rate = rate * ref_rate

            p_road = pred == 2
            p_road = p_road * 1.0                       # pred road detect

            ref_water_mask = water_ref_pred[:,:] == 1
            ref_road_mask = road_ref_pred[:,:] == 1
            ref_result[2,ref_water_mask] = 255
            ref_result[1,ref_road_mask] = 255

            ref_org_img = ref_org_img.transpose(2,0,1)
            ref_result = ref_result * b + ref_org_img * weight


            onroad_mask = (p_water[:,:] == 1) & (road_ref_pred[:,:] == 1)
            water_mask = (p_water[:,:] == 1) & (road_ref_pred[:,:] == 0)
            road_mask = p_road[:,:] == 1

            org_img = org_img.transpose(2,0,1)
            result[0,onroad_mask] = 255
            result[2,water_mask] = 255
            result[1,road_mask] = 255
            result = result*b + org_img*weight

            result = result.astype(dtype=np.uint8)
            ref_result = ref_result.astype(dtype=np.uint8)

            ref_result = Image.fromarray(ref_result.transpose(1,2,0))
            ref_result.save("./result/"+"ref_"+img_name+".png")

            result = Image.fromarray(result.transpose(1,2,0))
            result.save("./result/"+img_name+".png")
            RoadIoU = get_IoU(road, p_road)
            WaterIoU = get_IoU(water, p_water)
        return WaterIoU, RoadIoU, inundation_rate, base_rate

    elif model_name == "Vnet" or model_name == "3DUnet":

        gt_dir = "./gt"
        vid_dir = "./flood"
        ref_dir = "./ref"
        frm_ch_num = 16
        frm_period = 5
        data_height = 320
        data_width = 576
        with torch.no_grad():
            if model_name == "Vnet":
                model = vnet_wr.VNet(elu=True, nll=False, frm_ch=frm_ch_num, height=data_height, width=data_width)
                model.load_state_dict(torch.load("./model/" + model_name + ".pth")['model_state_dict'])
            elif model_name == "3DUnet":
                model = unet_2D3D.UNet3D(3,3)
                model.load_state_dict(torch.load("./model/" + model_name + ".pth")['model_state_dict'])
            model.eval()
            inputs, gt_w, gt_r = get_input(gt_dir, vid_dir, img_name, data_height, data_width, frm_ch_num, frm_period)
            ref_inputs, _, _ = get_input(gt_dir, ref_dir, img_name, data_height, data_width, frm_ch_num, frm_period)

            if torch.cuda.is_available():
                inputs = inputs.to('cuda')
                ref_inputs = ref_inputs.to('cuda')
                model.to('cuda')
            ref_outputs = model(ref_inputs)
            outputs = model(inputs)
            if model_name == "Vnet":
                outputs = outputs.cpu().numpy().reshape(frm_ch_num, data_height, data_width, 3)  # N F H W C
                ref_outputs = ref_outputs.cpu().numpy().reshape(frm_ch_num, data_height, data_width, 3)
            elif model_name == "3DUnet":
                outputs = outputs.cpu().numpy()
                ref_outputs = ref_outputs.cpu().numpy()
                outputs = np.squeeze(outputs)
                ref_outputs = np.squeeze(ref_outputs)
                outputs = np.transpose(outputs, (1, 2, 3, 0))
                ref_outputs = np.transpose(ref_outputs, (1, 2, 3, 0))
            inputs = inputs.cpu().numpy().squeeze(0) * 255
            inputs = inputs.astype(dtype=np.uint8)
            ref_inputs = ref_inputs.cpu().numpy().squeeze(0) * 255
            ref_inputs = ref_inputs.astype(dtype=np.uint8)
        wIoU = 0
        rIoU = 0
        inundation_rate = 0
        rates = 0

        for num in range(frm_ch_num):
            rref_outputs = ref_outputs[num,:,:]
            rref_outputs = rref_outputs.argmax(axis = 2)
            road_mask = rref_outputs == 1
            water_mask = rref_outputs == 0

            input = inputs[:, num, :, :]
            rref_inputs = ref_inputs[:,num,:,:]
            w_pred, r_pred, water_1 = get_img(num, outputs, rref_inputs, input, img_name, road_mask, water_mask)
            w_pred = w_pred * 1.0
            r_pred = r_pred * 1.0
            wIoU = wIoU + get_IoU(w_pred, gt_w[:, 0, :, :].numpy())
            rIoU = rIoU + get_IoU(r_pred, gt_r[:, 0, :, :].numpy())
            road_mask = road_mask * 1.0
            inundation_rates, rate = get_road_inundation(water_1, road_mask, water_mask)
            inundation_rate = inundation_rate + inundation_rates
            rates = rates + rate

        wIoU = wIoU / frm_ch_num
        rIoU = rIoU / frm_ch_num
        inundation_rate = inundation_rate / frm_ch_num
        base_rate = rates * ref_rate/frm_ch_num
        torch.cuda.empty_cache()

        return wIoU, rIoU, inundation_rate, base_rate

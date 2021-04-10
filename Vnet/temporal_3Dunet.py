from torch.utils.tensorboard import SummaryWriter
import torch
from default_set import dataset,loss_func
import torch.optim as optim
from torch.optim import lr_scheduler

import os
from default_set.unet_2D3D import *
from temporal_ex.unet_train_val import *
# ================ 3D U Net ===========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

set_frm_ch_num = 16
set_frm_period = 5
set_batch_size = 2
set_base_lr = 0.0006  # scheduler setting
set_max_lr = 0.0012  # scheduler setting
set_step_size_up = 200  # scheduler setting
set_step_size_down = 200

set_wt_save_name = '3DUnet_diceL1_fullsz_test_191112_1'
where = "server"  # "home" #"lab"
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
    set_gtdir = '/datahdd/WaterDetection/final_water_video_dataset/Train/annot/'
    set_videodir = "/datahdd/WaterDetection/final_water_video_dataset/Train/frames/"
    set_gtdir2 = '/datahdd/WaterDetection/final_water_video_dataset/Test/annot/'
    set_videodir2 = "/datahdd/WaterDetection/final_water_video_dataset/Test/frames/"
    set_wt_save_path = "/datahdd/WaterDetection/save_model/unet3D/"

else:
    raise Exception("Input 'where'.")

#------------------------------------------
if not os.path.exists(set_wt_save_path):
    os.makedirs(set_wt_save_path)
    print("** make dir for saving model **")

print("==="*20)
print("Check before training")
gtdir =  set_gtdir
videodir = set_videodir
gtdir2 =  set_gtdir2
videodir2 = set_videodir2
print("Ground truth image dir : ", gtdir)
print("Frames dir : ", videodir)

print("---"*20)
print("Create dataset class")
frm_ch_num = set_frm_ch_num
frm_period = set_frm_period
print("# of frames : ", frm_ch_num)
print("frame period : ", frm_period)
trainset = dataset.Dataset3D(gtdir,videodir,frm_ch_num,frm_period)
valset = dataset.Dataset3D(gtdir2,videodir2,frm_ch_num,frm_period)
print("Dataset length : ",trainset.__len__())
frms, gt_w, gt_r = trainset[0] #,gt_r
print("Frms shape : ",frms.shape)
print("Water ground truth shape : ", gt_w.shape)
print("Road ground truth shape : ", gt_r.shape)
data_height = 320
data_width = 576

print("-"*20)
print("Create dataloader")
batch_sz = set_batch_size
print("Batch size : ",batch_sz)
trainloader =torch.utils.data.DataLoader(trainset, batch_size = batch_sz, shuffle= True, num_workers=8)
valloader =torch.utils.data.DataLoader(valset, batch_size = batch_sz, shuffle= True, num_workers=8)
videos, gts_w, gts_r  = next(iter(trainloader)) #, gts_r , s_r
print("Videos shape : ",videos.shape)
print("Water gts shape : ",gts_w.shape)
print("Road gts shape : ",gts_r.shape)
#print("test v w r : ", s_v.shape, s_w.shape, s_r.shape) #, s_r.shape)

print("-"*20)
print("Create model")

model= UNet3D(3,3)

weight_decay = 1e-4
start_epoch = 0
best_mIoU = 0.0

if os.path.isfile(set_wt_save_path+set_wt_save_name+'_best.pth'):
    print('**********Resume*************')
    checkpoint = torch.load(set_wt_save_path+set_wt_save_name+'_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=set_base_lr, max_lr=set_max_lr,
                                      step_size_up=set_step_size_up, cycle_momentum=False)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch']
    best_mIoU = checkpoint['best_mIoU']
    best_w_mIoU = checkpoint['best_w_mIoU']
    best_r_mIoU = checkpoint['best_r_mIoU']
    print("< save point >")
    print("epoch : ", start_epoch)
    print("Best mIoU : ", best_mIoU)
    print("Best w mIoU : ", best_w_mIoU)
    print("Best r mIoU : ", best_r_mIoU)
else :
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=set_base_lr, max_lr=set_max_lr,
                                                  step_size_up=set_step_size_up, cycle_momentum=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device type:", device.type)
if torch.cuda.device_count() > 1:
    print("Use",torch.cuda.device_count(),"GPUs")
    model = nn.DataParallel(model)

model.to(device)
num_epochs = 10000

print("="*30)
print("Start train")
train_3Dunet(model,optimizer,scheduler,trainloader,valloader,
           set_wt_save_path,set_wt_save_name,device,best_mIoU,start_epoch,num_epochs)

#train_unet()

#train_3dunet()
from torch.utils.tensorboard import SummaryWriter
import torch
from default_set import dataset,loss_func

from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim import lr_scheduler

import os
from default_set.unet_2D3D import *
from temporal_ex.unet_train_val import train_2Dunet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#=======================================================================
gtdir = '/datahdd/WaterDetection/final_water_video_dataset/2d_dataset/Train/annot/'
imgdir = '/datahdd/WaterDetection/final_water_video_dataset/2d_dataset/Train/image/'
gtdir2 = '/datahdd/WaterDetection/final_water_video_dataset/2d_dataset/Test/annot/'
imgdir2 = '/datahdd/WaterDetection/final_water_video_dataset/2d_dataset/Test/image/'

set_batch_size = 11
set_base_lr = 0.0004  # scheduler setting
set_max_lr = 0.0012  # scheduler setting
set_step_size_up = 500  # scheduler setting
weight_decay = 1e-4

set_wt_save_path = '/datahdd/WaterDetection/save_model/unet2D/'
set_wt_save_name = 'unet_dicel1_fllsz_bch11_191113_1'
#=======================================================================
if not os.path.exists(set_wt_save_path):
    os.makedirs(set_wt_save_path)
    print("** make dir for saving model **")


model = UNet2D(3,3)
if torch.cuda.device_count() > 1:
    print("Use",torch.cuda.device_count(),"GPUs")
    model = nn.DataParallel(model)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

best_mIoU = 0.0
start_epoch = 0

trainset = dataset.Dataset2D(gtdir,imgdir)
valset = dataset.Dataset2D(gtdir2,imgdir2)

trainloader = DataLoader(trainset,batch_size=set_batch_size,shuffle=True)
valloader = DataLoader(valset,batch_size=set_batch_size,shuffle=True)



if os.path.isfile(set_wt_save_path+set_wt_save_name+'_best.pth'):
    print('**********Resume*************')
    checkpoint = torch.load(set_wt_save_path+set_wt_save_name+'_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()

    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=set_base_lr, max_lr=set_max_lr,
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

num_epochs = 10000
best_epoch_loss = 1.0
best_epoch_loss_w = 1.0
best_epoch_loss_r = 1.0

best_water_loss = 1.0
best_road_loss = 1.0




print("=" * 50)
print("Start Train")
print("=" * 50)
train_2Dunet(model,optimizer,scheduler,trainloader,valloader,
             set_wt_save_path,set_wt_save_name,device,
             best_mIoU,start_epoch,num_epochs)
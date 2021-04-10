from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable

import torch

import cv2
import os
import time

from default_set.unet_models import UNet

class water_unet_dataset(Dataset):
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

    def __getitem__(self,idx):
        #img = Image.open(self.imgdir+self.imglist[idx])
        #gt = Image.open(self.gtdir+self.gtlist[idx])
        img = cv2.imread(self.imgdir+self.imglist[idx])
        gt = cv2.imread(self.gtdir+self.gtlist[idx],cv2.IMREAD_GRAYSCALE)

        height = 16 * 3 * 5
        width = 16 * 4 * 5

        img = cv2.resize(img,(width,height))
        gt = cv2.resize(gt,(width,height),cv2.INTER_NEAREST)

        gt_w = (gt==255)*1.0
        gt_r = (gt==125)*1.0

        img = self.transform(img)
        gt_w = self.totnsr(gt_w)
        gt_r = self.totnsr(gt_r)

        return img, gt_w, gt_r

def dice_loss(pred, target):
    smooth = 1.
 
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
 
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
 
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def L1_loss(pred, target):
    f_pred = pred.contiguous().view(-1)
    f_target = target.contiguous().view(-1)
    l1loss = (torch.abs(f_pred-f_target).sum())/f_pred.shape[0]
    return l1loss

os.environ["CUDA_VISIBLE_DEVICES"]="2"

#model = unet11(pretrained=True) #classes 3 (water, road , none)
model = UNet(3)
#=======================================================================
gtdir = '/datahdd/WaterDetection/water_video_dataset/2dconv_dataset/Train/annot/'
imgdir = '/datahdd/WaterDetection/water_video_dataset/2dconv_dataset/Train/image/'
set_batch_size = 7
set_base_lr = 0.0002  # scheduler setting
set_max_lr = 0.0012  # scheduler setting
set_step_size_up = 200  # scheduler setting
weight_decay = 1e-4

set_wt_save_path = '/home/hyeongeun/model_save/'
set_wt_save_name = 'unet_dicel1_wr_bch7_191107'
#=======================================================================

dataset = water_unet_dataset(gtdir,imgdir)

trainloader = DataLoader(dataset,batch_size=set_batch_size,shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr = set_base_lr, max_lr=set_max_lr,
                                  step_size_up = set_step_size_up, cycle_momentum=False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.train()
epochs = 10000
best_epoch_loss = 1.0
best_epoch_loss_w = 1.0
best_epoch_loss_r = 1.0
 
best_water_loss = 1.0
best_road_loss = 1.0
print("="*50)
print("Start Train")
print("="*50)
for epoch in range(epochs):
  epoch_start_time =time.time()
  epoch_loss = 0.0
  water_epoch_loss = 0.0
  road_epoch_loss = 0.0
  for batch_idx, (frms, gts_w, gts_r) in enumerate(trainloader):
    frms, gts_w, gts_r = frms.cuda(), gts_w.cuda(),gts_r.cuda()#, gts_r.cuda()#, gts_r
    frms, gts_w, gts_r = Variable(frms), Variable(gts_w), Variable(gts_r)#, gts_r, Variable(gts_r)
    optimizer.zero_grad()
    #print("frms, gtw, gtr : ",frms.shape,gts_w.shape,gts_r.shape)
    output = model(frms)
    #print("output.shape : ",output.shape)
    w_di = dice_loss(output[:,0,:,:],gts_w)
    w_l1 = L1_loss(output[:,0,:,:],gts_w)
    water_loss = (w_di+w_l1)/2

    r_di = dice_loss(output[:,1,:,:], gts_r)
    r_l1 = L1_loss(output[:, 1, :, :], gts_r)
    road_loss =(r_di+r_l1)/2

    loss = water_loss + road_loss
    loss.backward()
    optimizer.step()
 
    epoch_loss += loss
    water_epoch_loss += water_loss
    road_epoch_loss += road_loss
  epoch_loss /= len(trainloader)
  water_epoch_loss /= len(trainloader)
  road_epoch_loss /= len(trainloader)
  
  epoch_time = time.time() - epoch_start_time
  time_h = int((epoch_time // 60) // 60)
  time_m = int((epoch_time // 60) % 60)
  time_s = epoch_time % 60

  if best_epoch_loss>epoch_loss:
    best_epoch_loss = epoch_loss
    best_epoch_loss_w = water_epoch_loss
    best_epoch_loss_r = road_epoch_loss
 
    wt_save_path = set_wt_save_path
    wt_save_name = set_wt_save_name + '_best.pth'
    torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_epoch_loss':best_epoch_loss,
            'water_epoch_loss':best_epoch_loss_w,
            'road_epoch_loss':best_epoch_loss_r
        }, wt_save_path+wt_save_name)
 
  wt_save_path = set_wt_save_path
  wt_save_name = set_wt_save_name +'_last.pth'
  torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_epoch_loss': best_epoch_loss,
        'water_epoch_loss': water_epoch_loss,
        'road_epoch_loss': road_epoch_loss
    }, wt_save_path + wt_save_name)
  print("Epoch {}/{} Total loss : {:.8f} ( Best total loss : {:.8f} = dice_w {:.8f} + dice_r {:.8f} , lr = {} ,time = {:.0f}m {:.0f}s )".format(epoch, epochs - 1, epoch_loss, best_epoch_loss,
                                                                                                                        best_epoch_loss_w, best_epoch_loss_r,
                                                                                                                        scheduler.get_lr(),
                                                                                                                        time_m,time_s))
  scheduler.step()









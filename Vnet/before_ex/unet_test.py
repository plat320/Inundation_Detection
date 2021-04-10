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

    def __getitem__(self, idx):
        # img = Image.open(self.imgdir+self.imglist[idx])
        # gt = Image.open(self.gtdir+self.gtlist[idx])
        img = cv2.imread(self.imgdir + self.imglist[idx])
        gt = cv2.imread(self.gtdir + self.gtlist[idx], cv2.IMREAD_GRAYSCALE)

        height = 16 * 3 * 5
        width = 16 * 4 * 5

        img = cv2.resize(img, (width, height))
        gt = cv2.resize(gt, (width, height), cv2.INTER_NEAREST)

        gt_w = (gt == 255) * 1.0
        gt_r = (gt == 125) * 1.0

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

def iou_cal(pred, target):
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return intersection / (A_sum + B_sum - intersection)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#model = unet11(pretrained=True)  # classes 3 (water, road , none)
model = UNet(3)
# =======================================================================
gtdir = '/datahdd/WaterDetection/water_video_dataset/2dconv_dataset/Test/annot/'
imgdir = '/datahdd/WaterDetection/water_video_dataset/2dconv_dataset/Test/image/'
set_batch_size = 16
#set_base_lr = 0.0005  # scheduler setting
#set_max_lr = 0.01  # scheduler setting
#set_step_size_up = 800  # scheduler setting
#weight_decay = 1e-4

set_model_load_path = '/home/hyeongeun/model_save/'
set_model_name = "unet_dicel1_wr_bch7_191107_best.pth"

set_save_result_img_path = '/home/hyeongeun/model_test/'+set_model_name[:-4]+"/"
# =======================================================================

dataset = water_unet_dataset(gtdir, imgdir)

testloader = DataLoader(dataset, batch_size=set_batch_size, shuffle=False)

model_load_path= set_model_load_path + set_model_name
checkpoint = torch.load(model_load_path)

model.load_state_dict(checkpoint['model_state_dict'])
print("< save point >")
print("Epoch : ",checkpoint['epoch'])
print("Best epoch loss : ",checkpoint['best_epoch_loss'])
print("Best Water epoch loss : ",checkpoint['water_epoch_loss'])
print("Best Road epoch loss : ",checkpoint['road_epoch_loss'])

#'epoch': epoch,
#'model_state_dict': model.state_dict(),
#'optimizer_state_dict': optimizer.state_dict(),
#'best_epoch_loss': best_epoch_loss,
#'water_epoch_loss': water_epoch_loss,
#'road_epoch_loss': road_epoch_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()
epochs = 10000
best_epoch_loss = 1.0
best_epoch_loss_w = 1.0
best_epoch_loss_r = 1.0

best_water_loss = 1.0
best_road_loss = 1.0
print("=" * 50)
print("Start Test")
print("=" * 50)
with torch.no_grad():
    cnt = 0
    epoch_start_time = time.time()
    epoch_loss = 0.0
    water_epoch_loss = 0.0
    road_epoch_loss = 0.0

    t_water_iou = 0
    t_road_iou = 0
    for frms, gts_w, gts_r in testloader:

        frms, gts_w, gts_r = frms.cuda(), gts_w.cuda(), gts_r.cuda()  # , gts_r.cuda()#, gts_r
        frms, gts_w, gts_r = Variable(frms), Variable(gts_w), Variable(gts_r)  # , gts_r, Variable(gts_r)

        output = model(frms)

        water_loss = dice_loss(output[:, 0, :, :], gts_w)
        road_loss = dice_loss(output[:, 1, :, :], gts_r)
        loss = water_loss + road_loss

        water_iou = iou_cal(output[:, 0, :, :], gts_w)
        road_iou = iou_cal(output[:, 1, :, :], gts_r)

        waters_pred = (output[:,0,:,:] > 0.5)*1
        roads_pred = (output[:,1,:,:] > 0.5)*1

        dirpath = set_save_result_img_path
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        print("batch size ",waters_pred.shape[0])
        for i in range(waters_pred.shape[0]):
            w_p_ = waters_pred[i,:,:]
            r_p_ = roads_pred[i,:,:]

            img_ = frms[i,:,:,:]*255
            img_[0, :, :] = (1 - w_p_) * img_[0, :, :] + (w_p_ * 255)
            img_[1, :, :] = (1 - w_p_) * img_[1, :, :]
            img_[2, :, :] = (1 - w_p_) * img_[2, :, :]

            img_[0, :, :] = (1 - r_p_) * img_[0, :, :]
            img_[1, :, :] = (1 - r_p_) * img_[1, :, :] + (r_p_ * 255)
            img_[2, :, :] = (1 - r_p_) * img_[2, :, :]

            img_ = img_.permute(1,2,0)

            w_p_ = w_p_*255
            r_p_ = r_p_ * 255

            w_p = w_p_.cpu().detach().numpy()
            r_p = r_p_.cpu().detach().numpy()
            img = img_.cpu().detach().numpy()

            num_name = cnt * set_batch_size + i + 1
            w_p_path = dirpath + "water_%03d.jpg" % num_name
            r_p_path = dirpath + "road_%03d.jpg" % num_name
            img_path = dirpath + "overimg_%03d.jpg" % num_name

            cv2.imwrite(w_p_path, w_p)
            cv2.imwrite(r_p_path, r_p)
            cv2.imwrite(img_path, img)

        cnt += 1
        epoch_loss += loss
        water_epoch_loss += water_loss
        road_epoch_loss += road_loss

        t_water_iou += water_iou
        t_road_iou += road_iou
        print(cnt, "-- Water mIoU {:.5f}, Road mIoU {:.5f}".format(water_iou, road_iou))

    epoch_loss /= len(testloader)
    water_epoch_loss /= len(testloader)
    road_epoch_loss /= len(testloader)

    water_miou = t_water_iou/len(testloader)
    road_miou = t_road_iou / len(testloader)

    epoch_time = time.time() - epoch_start_time
    time_h = int((epoch_time // 60) // 60)
    time_m = int((epoch_time // 60) % 60)
    time_s = epoch_time % 60
    print("<<<  Test result  >>>")
    print("Total loss {:.5f}".format(epoch_loss))
    print("Water D loss {:.5f}, Road D loss {:.5f}".format(water_epoch_loss,road_epoch_loss))
    print("Water mIoU {:.5f}, Road mIoU {:.5f}".format(water_miou,road_miou))
    print("time {:.0f}m {:.0f}s".format(time_m,time_s))








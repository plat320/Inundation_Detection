from __future__ import absolute_import

import torch
import os
import time
from default_set import loss_func

def val_vnet(model, optimizer, scheduler,valloader,
             epoch, set_wt_save_path,set_wt_save_name,
             best_mIoU, device):
    epoch_start_time = time.time()
    print("=" * 20)
    print("<start validation>")
    model.eval()
    with torch.no_grad():
        Total_w_mIoU = 0.0
        Total_r_mIoU = 0.0
        Total_max_w_mIoU = 0.0
        Total_max_r_mIoU = 0.0
        for idx, (frms, gts_w, gts_r) in enumerate(valloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)
            output = model(frms)
            pred_water = output[:, :, 0]
            pred_road = output[:, :, 1]

            w_pred = pred_water.contiguous().view(16, 320, 576)
            w_pred_thr = (w_pred > 0.5) * 1.0

            r_pred = pred_road.contiguous().view(16, 320, 576)
            r_pred_thr = (r_pred > 0.5) * 1.0

            w_mIoU = loss_func.iou_cal(w_pred_thr, gts_w)
            r_mIoU = loss_func.iou_cal(r_pred_thr, gts_r)

            Total_w_mIoU += w_mIoU
            Total_r_mIoU += r_mIoU
            #print(idx,' --- w_mIoU :', w_mIoU, ' / r_mIoU :', r_mIoU)

            max_w_mIoU = 0
            max_r_mIoU = 0

            for i in range(16):
                wpred_thr = w_pred_thr[i, :, :]
                rpred_thr = r_pred_thr[i, :, :]
                if max_w_mIoU < loss_func.iou_cal(wpred_thr, gts_w[:, :, i, :, :]):
                    max_w_mIoU = loss_func.iou_cal(wpred_thr, gts_w[:, :, i, :, :])
                if max_r_mIoU < loss_func.iou_cal(rpred_thr, gts_r[:, :, i, :, :]):
                    max_r_mIoU = loss_func.iou_cal(rpred_thr, gts_r[:, :, i, :, :])

            print('max_w_mIoU : ', max_w_mIoU, ' / max_r_mIoU : ', max_r_mIoU)
            Total_max_w_mIoU += max_w_mIoU
            Total_max_r_mIoU += max_r_mIoU
        Total_w_mIoU /= len(valloader)
        Total_r_mIoU /= len(valloader)
        Total_max_w_mIoU /= len(valloader)
        Total_max_r_mIoU /= len(valloader)
        print("Val result : Total_w_mIoU = {:.5f}, Total_r_mIoU = {:.5f}".format(Total_w_mIoU, Total_r_mIoU))
        print("Val result : Total_max_w_mIoU = {:.5f}, Total_max_r_mIoU = {:.5f}".format(Total_max_w_mIoU,
                                                                                          Total_max_r_mIoU))
        if best_mIoU < (Total_w_mIoU+Total_r_mIoU)/2 :
            best_mIoU = (Total_w_mIoU+Total_r_mIoU)/2
            wt_save_path = set_wt_save_path
            wt_save_name = set_wt_save_name + '_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mIoU': best_mIoU,
                'best_w_mIoU': Total_w_mIoU,
                'best_r_mIoU': Total_r_mIoU
            }, wt_save_path + wt_save_name)
            print('**** Save new best.pth ****')

    epoch_time = time.time() - epoch_start_time
    time_m = int((epoch_time // 60) % 60)
    time_s = epoch_time % 60
    print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
    print("="*20)



def train_vnet(model, optimizer, scheduler, trainloader, valloader,
               set_wt_save_path,set_wt_save_name, device,
               best_mIoU, start_epochs, num_epochs):
    best_epoch_loss = 1.0
    best_epoch_loss_w = 1.0
    best_epoch_loss_r = 1.0
    model.train()
    for epoch in range(start_epochs,num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*10)
        epoch_loss = 0.0
        water_epoch_loss = 0.0
        road_epoch_loss = 0.0

        for batch_idx, (frms, gts_w, gts_r) in enumerate(trainloader):  # , gts_r, show_gts_r
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)

            optimizer.zero_grad()
            output = model(frms)

            water_loss = (loss_func.dice_loss(output[:, :, 0], gts_w) + loss_func.L1_loss(output[:, :, 0], gts_w)) / 2  # ,batch_sz)
            road_loss = (loss_func.dice_loss(output[:, :, 1], gts_r) + loss_func.L1_loss(output[:, :, 1], gts_r)) / 2  # ,batch_sz)

            loss = (water_loss + road_loss)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            water_epoch_loss += water_loss.item()
            road_epoch_loss += road_loss.item()

        epoch_loss /= len(trainloader)
        water_epoch_loss /= len(trainloader)
        road_epoch_loss /= len(trainloader)
        scheduler.step()

        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch_loss_w = water_epoch_loss
            best_epoch_loss_r = road_epoch_loss

        if epoch % 5 == 0:
            # save last epoch
            wt_save_path = set_wt_save_path
            wt_save_name = set_wt_save_name + '_last.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch_loss': epoch_loss,
                'water_epoch_loss': water_epoch_loss,
                'road_epoch_loss': road_epoch_loss,

            }, wt_save_path + wt_save_name)

            # save best epoch
            val_vnet(model, optimizer, scheduler,valloader,
                     epoch, set_wt_save_path,set_wt_save_name,
                     best_mIoU, device)

        print("Total loss : {:.8f} ( Best total loss : {:.8f} = focal dice+l1_w {:.8f} + focal dice+l1_r {:.8f} , lr = {} )".format(
                 epoch_loss, best_epoch_loss,
                best_epoch_loss_w, best_epoch_loss_r,
                scheduler.get_lr()))
        epoch_time = time.time() - since
        time_m = int((epoch_time // 60) % 60)
        time_s = epoch_time % 60
        print("Train epoch time : {:.0f}min {:.1f}sec".format(time_m, time_s))
from __future__ import absolute_import

import torch
import os
import time
from default_set.loss_func import *

def val_2Dunet(model, optimizer, scheduler, valloader,
                     epoch, set_wt_save_path, set_wt_save_name,
                     best_mIoU, device):
    epoch_start_time = time.time()
    print("=" * 20)
    print("<start validation>")

    model.eval()
    with torch.no_grad():
        Total_w_mIoU = 0.0
        Total_r_mIoU = 0.0
        for idx, (frms, gts_w, gts_r) in enumerate(valloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)

            output = model(frms)

            water_pred = (output[:, 0, :, :] > 0.5) * 1
            road_pred = (output[:, 1, :, :] > 0.5) * 1

            w_mIoU = iou_cal(water_pred, gts_w)
            r_mIoU = iou_cal(road_pred, gts_r)

            Total_w_mIoU += w_mIoU
            Total_r_mIoU += r_mIoU

        Total_w_mIoU /= len(valloader)
        Total_r_mIoU /= len(valloader)
        print("Val result : Total_w_mIoU = {:.5f}, Total_r_mIoU = {:.5f}".format(Total_w_mIoU, Total_r_mIoU))

        if ( best_mIoU < (Total_w_mIoU+Total_r_mIoU)/2 ) :
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
            print("Best mIoU = {:.5f}".format(best_mIoU))
            epoch_time = time.time() - epoch_start_time
            time_m = int((epoch_time // 60) % 60)
            time_s = epoch_time % 60
            print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
            print("=" * 20)
            return best_mIoU
        else :
            print("Best mIoU = {:.5f}".format(best_mIoU))
            epoch_time = time.time() - epoch_start_time
            time_m = int((epoch_time // 60) % 60)
            time_s = epoch_time % 60
            print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
            print("=" * 20)
            return best_mIoU


def train_2Dunet(model,optimizer,scheduler,trainloader,valloader,
           set_wt_save_path,set_wt_save_name,device,
           best_mIoU,start_epoch,num_epochs):
    best_epoch_loss = 4.0
    model.train()
    for epoch in range(start_epoch,num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        water_epoch_loss = 0.0
        road_epoch_loss = 0.0

        if epoch % 5 == 0 :
            wt_save_path = set_wt_save_path
            wt_save_name = set_wt_save_name + '_last.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, wt_save_path + wt_save_name)

            best_mIoU = val_2Dunet(model, optimizer, scheduler, valloader,
                     epoch, set_wt_save_path, set_wt_save_name,
                     best_mIoU, device)

        for batch_idx, (frms, gts_w, gts_r) in enumerate(trainloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)  # , gts_r.cuda()#, gts_r
            optimizer.zero_grad()
            output = model(frms)

            w_di = dice_loss(output[:, 0, :, :], gts_w)
            w_l1 = L1_loss(output[:, 0, :, :], gts_w)
            water_loss = (w_di + w_l1)

            r_di = dice_loss(output[:, 1, :, :], gts_r)
            r_l1 = L1_loss(output[:, 1, :, :], gts_r)
            road_loss = (r_di + r_l1)

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

        time_m = int((epoch_time // 60) % 60)
        time_s = epoch_time % 60

        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch_loss_w = water_epoch_loss
            best_epoch_loss_r = road_epoch_loss

        print(
            "Epoch {}/{} Total loss : {:.8f} ( Best total loss : {:.8f} = dice_w {:.8f} + dice_r {:.8f} , lr = {} ,time = {:.0f}m {:.0f}s )".format(
                epoch, num_epochs - 1, epoch_loss, best_epoch_loss,
                best_epoch_loss_w, best_epoch_loss_r,
                scheduler.get_lr(),
                time_m, time_s))
        scheduler.step()


#======3D UNET=====================================================================================================================

def val_3Dunet(model, optimizer, scheduler, valloader,
                     epoch, set_wt_save_path, set_wt_save_name,
                     best_mIoU, device):
    epoch_start_time = time.time()
    print("=" * 20)
    print("<start validation>")

    model.eval()
    with torch.no_grad():
        Total_w_mIoU = 0.0
        Total_r_mIoU = 0.0
        for idx, (frms, gts_w, gts_r) in enumerate(valloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)

            output = model(frms)

            water_pred = (output[:, 0, :, :, :] > 0.5) * 1
            road_pred = (output[:, 1, :, :, :] > 0.5) * 1

            w_mIoU = iou_cal(water_pred, gts_w)
            r_mIoU = iou_cal(road_pred, gts_r)

            Total_w_mIoU += w_mIoU
            Total_r_mIoU += r_mIoU

        Total_w_mIoU /= len(valloader)
        Total_r_mIoU /= len(valloader)
        print("Val result : Total_w_mIoU = {:.5f}, Total_r_mIoU = {:.5f}".format(Total_w_mIoU, Total_r_mIoU))

        if ( best_mIoU < (Total_w_mIoU+Total_r_mIoU)/2 ) :
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
            print("Best mIoU = {:.5f}".format(best_mIoU))
            epoch_time = time.time() - epoch_start_time
            time_m = int((epoch_time // 60) % 60)
            time_s = epoch_time % 60
            print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
            print("=" * 20)
            return best_mIoU
        else :
            print("Best mIoU = {:.5f}".format(best_mIoU))
            epoch_time = time.time() - epoch_start_time
            time_m = int((epoch_time // 60) % 60)
            time_s = epoch_time % 60
            print("Val Time : {:.0f}min {:.1f}sec".format(time_m, time_s))
            print("=" * 20)
            return best_mIoU


def train_3Dunet(model,optimizer,scheduler,trainloader,valloader,
           set_wt_save_path,set_wt_save_name,device,
           best_mIoU,start_epoch,num_epochs):
    best_epoch_loss = 4.0
    model.train()
    for epoch in range(start_epoch,num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        water_epoch_loss = 0.0
        road_epoch_loss = 0.0

        if epoch % 5 == 0 :
            wt_save_path = set_wt_save_path
            wt_save_name = set_wt_save_name + '_last.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, wt_save_path + wt_save_name)

            best_mIoU = val_3Dunet(model, optimizer, scheduler, valloader,
                     epoch, set_wt_save_path, set_wt_save_name,
                     best_mIoU, device)

        for batch_idx, (frms, gts_w, gts_r) in enumerate(trainloader):
            frms, gts_w, gts_r = frms.to(device), gts_w.to(device), gts_r.to(device)  # , gts_r.cuda()#, gts_r
            optimizer.zero_grad()
            output = model(frms)

            w_di = dice_loss(output[:, 0, :, :], gts_w)
            w_l1 = L1_loss(output[:, 0, :, :], gts_w)
            water_loss = (w_di + w_l1)

            r_di = dice_loss(output[:, 1, :, :], gts_r)
            r_l1 = L1_loss(output[:, 1, :, :], gts_r)
            road_loss = (r_di + r_l1)

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

        time_m = int((epoch_time // 60) % 60)
        time_s = epoch_time % 60

        if best_epoch_loss > epoch_loss:
            best_epoch_loss = epoch_loss
            best_epoch_loss_w = water_epoch_loss
            best_epoch_loss_r = road_epoch_loss

        print(
            "Epoch {}/{} Total loss : {:.8f} ( Best total loss : {:.8f} = dice_w {:.8f} + dice_r {:.8f} , lr = {} ,time = {:.0f}m {:.0f}s )".format(
                epoch, num_epochs - 1, epoch_loss, best_epoch_loss,
                best_epoch_loss_w, best_epoch_loss_r,
                scheduler.get_lr(),
                time_m, time_s))
        scheduler.step()

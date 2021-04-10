import torch
import torch.nn as nn

#dice, L1, diceL1, focal dice, focal diceL1

def iou_cal(pred, target):
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return intersection / (A_sum + B_sum - intersection)


def dice_loss(pred, target):
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1-((2. * intersection + smooth) / (A_sum + B_sum + smooth))

def L1_loss(pred, target):
    f_pred = pred.contiguous().view(-1)
    f_target = target.contiguous().view(-1)

    return torch.abs(f_pred-f_target).sum()/f_target.shape[0]

def diceL1_loss(pred, target):
    return (dice_loss(pred,target) + L1_loss(pred,target))/2

def focal_dice_loss(pred, target, batch_size, gamma=2):
    f_pred = pred.contiguous().view(batch_size, -1)
    f_target = target.contiguous().view(batch_size, -1)

    gt1_mask = f_target.contiguous()
    gt0_mask = f_target == 0

    pt_gt1 = f_pred * gt1_mask
    pt_gt0 = 1. * gt0_mask - f_pred * gt0_mask
    pt = pt_gt1 + pt_gt0
    pt = torch.sum(pt, 1) / f_target.shape[1]

    smooth = 1.
    inter = torch.sum(f_pred*f_target,1)
    p_sum = torch.sum(f_pred*f_pred,1)
    g_sum = torch.sum(f_target * f_target, 1)
    dice = 1-((2.*inter + smooth)/(p_sum + g_sum + smooth))

    dice_focal = ((1-pt)**gamma)*dice
    dice_focal = dice_focal.sum()/batch_size

    return dice_focal


def focal_diceL1_loss(pred, target, batch_size, gamma=2):
    f_pred = pred.contiguous().view(batch_size, -1)
    f_target = target.contiguous().view(batch_size, -1)
    gt1_mask = f_target.contiguous()
    gt0_mask = f_target == 0

    pt_gt1 = f_pred * gt1_mask
    pt_gt0 = 1. * gt0_mask - f_pred * gt0_mask
    pt = pt_gt1 + pt_gt0
    pt = torch.sum(pt, 1) / f_target.shape[1]

    smooth = 1.
    inter = torch.sum(f_pred * f_target, 1)
    p_sum = torch.sum(f_pred * f_pred, 1)
    g_sum = torch.sum(f_target * f_target, 1)
    dice = 1 - ((2. * inter + smooth) / (p_sum + g_sum + smooth))
    L1 = 1 - pt

    diceL1_focal = ((1 - pt) ** gamma) * (dice + L1)
    diceL1_focal = diceL1_focal.sum() / batch_size

    return diceL1_focal
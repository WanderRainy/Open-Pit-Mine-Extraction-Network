import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True, num_classes=1):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_true):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_true, y_pred)
        return a + b

class dice_bce_mse_loss(nn.Module):
    def __init__(self, batch=True, num_classes=1):
        super(dice_bce_mse_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    def __call__(self, y_pred, y_true):
        # print('label_pred:{}'.format(y_pred['label_pred'].size()))
        # print('label:{}'.format(y_true['label'].size()))
        # print(y_true.size())
        a =  self.bce_loss(y_pred['label_pred'], y_true['label'])
        b =  self.soft_dice_loss(y_pred['label_pred'], y_true['label'])
        c =  self.mse_loss(y_pred['height_pred'], y_true['height_label'])
        return a + b + c
# 用于多类
class dice_ce_loss(nn.Module):
    def __init__(self, batch=True, num_classes=3):
        super(dice_ce_loss, self).__init__()
        self.batch = batch
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss()

    def one_hot_encode(self, y_true, num_classes):
        # Convert (B, 1, H, W) to (B, C, H, W) where C is the number of classes
        y_true_one_hot = torch.zeros(y_true.size(0), num_classes, y_true.size(2), y_true.size(3)).to(y_true.device)
        y_true_one_hot.scatter_(1, y_true.long(), 1)
        return y_true_one_hot

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-3  # small value to avoid division by zero
        if self.batch:
            intersection = torch.sum(y_true * y_pred, dim=(0, 2, 3))
            i = torch.sum(y_true, dim=(0, 2, 3))
            j = torch.sum(y_pred, dim=(0, 2, 3))
        else:
            intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
            i = torch.sum(y_true, dim=(1, 2, 3))
            j = torch.sum(y_pred, dim=(1, 2, 3))
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        # import pdb
        # pdb.set_trace()
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W), y_true: (B, 1, H, W)
        y_true_one_hot = self.one_hot_encode(y_true, self.num_classes)  # Convert to (B, C, H, W)
        
        # Compute cross-entropy loss
        ce_loss = self.ce_loss(y_pred, y_true.squeeze(1).long())  # y_true needs to be (B, H, W) for CrossEntropyLoss
        
        # Apply softmax to y_pred for dice loss calculation
        y_pred_softmax = F.softmax(y_pred, dim=1)
        
        # Compute dice loss for each class
        dice_loss = self.soft_dice_loss(y_true_one_hot, y_pred_softmax)
        
        return ce_loss + dice_loss


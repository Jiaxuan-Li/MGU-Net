import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import  numpy as np


def loss_builder1():
    criterion_1_1 = nn.NLLLoss(ignore_index=255)
    criterion_1_2 = DiceLoss(class_num=3)
    criterion = [criterion_1_1,criterion_1_2]
    return criterion


def loss_builder2():
    criterion_2_1 = nn.NLLLoss(ignore_index=255)
    criterion_2_2 = DiceLoss(class_num=11)
    criterion = [criterion_2_1, criterion_2_2]
    return criterion

class DiceLoss(nn.Module):
    def __init__(self, class_num=11,smooth=1): 
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self,input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num - 1)
        return dice_loss
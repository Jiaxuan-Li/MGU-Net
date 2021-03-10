import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from models.nets.MGUNet import MGUNet_1, MGUNet_2 
from models.utils.utils import img2df, feature_fusion

class TSMGUNet(nn.Module):
    def __init__(self):  ##########
        super(TSMGUNet, self).__init__()
        self.stage1 = MGUNet_1(in_channels=1, n_classes=3, feature_scale=4)
        self.stage2 = MGUNet_2(in_channels=1, n_classes=10, feature_scale=4)
    
    def forward(self, inputs):
        input_layer2 = copy.deepcopy(inputs)
        out1 = self.stage1(inputs)
        output1 = F.log_softmax(out1, dim=1)
        _, pred = torch.max(output1, 1)
        input_mask = (pred).unsqueeze(1).float()
        input_disc_free = img2df(input_layer2,input_mask)
        out2 = self.stage2(input_disc_free)
        output2, out = feature_fusion(out1, out2)
        output = F.log_softmax(out, dim=1)
        return output1, output2, output
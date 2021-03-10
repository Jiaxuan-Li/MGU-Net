import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.utils import Basconv, UnetConv, UnetUp, UnetUp4, GloRe_Unit
from collections import OrderedDict
from models.utils.init_weights import init_weights


class  MGR_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MGR_Module, self).__init__()

        self.conv0_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou0 = nn.Sequential(OrderedDict([("GCN%02d" % i, GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv1_1 = Basconv(in_channels=in_channels,out_channels=out_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.conv1_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou1 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, out_channels, kernel=1)) for i in range(1)]))

        self.conv2_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.conv2_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou2 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))

        self.conv3_1 = Basconv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.conv3_2 = Basconv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.glou3 = nn.Sequential(OrderedDict([("GCN%02d" % i,GloRe_Unit(out_channels, int(out_channels/2), kernel=1)) for i in range(1)]))
        
        self.f1 = Basconv(in_channels=4*out_channels, out_channels=in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)

        self.x0 = self.conv0_1(x)
        self.g0 = self.glou0(self.x0)

        self.x1 = self.conv1_2(self.pool1(self.conv1_1(x)))
        self.g1 = self.glou1(self.x1)
        self.layer1 = F.interpolate(self.g1, size=(h, w), mode='bilinear', align_corners=True)

        self.x2 = self.conv2_2(self.pool2(self.conv2_1(x)))
        self.g2 = self.glou2(self.x2)
        self.layer2 = F.interpolate(self.g2, size=(h, w), mode='bilinear', align_corners=True)

        self.x3 = self.conv3_2(self.pool3(self.conv3_1(x)))
        self.g3= self.glou3(self.x3)
        self.layer3 = F.interpolate(self.g3, size=(h, w), mode='bilinear', align_corners=True)

        out = torch.cat([self.g0, self.layer1, self.layer2, self.layer3], 1)

        return self.f1(out)



class MGUNet_1(nn.Module):
    def __init__(self, in_channels=1, n_classes=11, feature_scale=4, is_deconv=True, is_batchnorm=True):  ##########
        super(MGUNet_1, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=4)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=4)

        self.mgb =  MGR_Module(filters[2], filters[3])

        self.center = UnetConv(filters[2], filters[3], self.is_batchnorm)

        # decoder
        self.up_concat3 = UnetUp4(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp4(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1)  
        conv2 = self.conv2(maxpool1)  
        maxpool2 = self.maxpool2(conv2)  
        conv3 = self.conv3(maxpool2) 
        maxpool3 = self.maxpool3(conv3)  
        feat_sum = self.mgb(maxpool3)
        center = self.center(feat_sum)  
        up3 = self.up_concat3(center, conv3) 
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)

        return final_1


class MGUNet_2(nn.Module):
    def __init__(self, in_channels=1, n_classes=11, feature_scale=4, is_deconv=True, is_batchnorm=True):  ##########
        super(MGUNet_2, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # encoder
        self.conv1 = UnetConv(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UnetConv(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UnetConv(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.mgb =  MGR_Module(filters[2], filters[3])

        self.center = UnetConv(filters[2], filters[3], self.is_batchnorm)

        # decoder
        self.up_concat3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp(filters[1], filters[0], self.is_deconv)

        # final conv
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  
        maxpool1 = self.maxpool1(conv1) 
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)  
        conv3 = self.conv3(maxpool2)  
        maxpool3 = self.maxpool3(conv3)  
        feat_sum = self.mgb(maxpool3) 
        center = self.center(feat_sum)  
        up3 = self.up_concat3(center, conv3) 
        up2 = self.up_concat2(up3, conv2) 
        up1 = self.up_concat1(up2, conv1)
        final_1 = self.final_1(up1)

        return final_1



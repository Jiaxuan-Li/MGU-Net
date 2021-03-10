import torch
import torch.nn as nn
import torch.nn.functional as F
from init_weights import init_weights

class Basconv(nn.Sequential):
    def __init__(self, in_channels, out_channels, is_batchnorm = False, kernel_size = 3, stride = 1, padding=1):
        super(Basconv, self).__init__()
        if is_batchnorm:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.BatchNorm2d(out_channels),nn.ReLU(inplace=True))
        else:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        x = inputs
        x = self.conv(x)
        return x

class UnetConv(nn.Module):
    def __init__(self, in_channels, out_channels, is_batchnorm, n=2, kernel_size = 3, stride=1, padding=1):
        super(UnetConv, self).__init__()
        self.n = n    

        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_channels = out_channels

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)
        return x

class UnetUp(nn.Module):
    def __init__(self,in_channels, out_channels, is_deconv, n_concat=2):
        super(UnetUp, self).__init__()
        self.conv = UnetConv(in_channels+(n_concat-2)* out_channels, out_channels, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0,*input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0,input[i]], 1)
        return self.conv(outputs0)


class UnetUp4(nn.Module):
    def __init__(self,in_channels, out_channels, is_deconv, n_concat=2):
        super(UnetUp4, self).__init__()
        self.conv = UnetConv(in_channels+(n_concat-2)* out_channels, out_channels, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=6, stride=4, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0,*input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0,input[i]], 1)
        return self.conv(outputs0)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        h = self.relu(h)
        h = self.conv2(h)
        return h

class GloRe_Unit(nn.Module):

    def __init__(self, num_in, num_mid, stride=(1,1), kernel=1):
        super(GloRe_Unit, self).__init__()

        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        kernel_size = (kernel, kernel)
        padding = (1, 1) if kernel == 3 else (0, 0)
        # reduce dimension
        self.conv_state = Basconv(num_in, self.num_s, is_batchnorm = True, kernel_size=kernel_size, padding=padding)  
        # generate projection and inverse projection functions
        self.conv_proj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)   
        self.conv_reproj = Basconv(num_in, self.num_n, is_batchnorm = True,kernel_size=kernel_size, padding=padding)  
        # reasoning by graph convolution
        self.gcn1 = GCN(num_state=self.num_s, num_node=self.num_n)   
        self.gcn2 = GCN(num_state=self.num_s, num_node=self.num_n)  
        # fusion
        self.fc_2 = nn.Conv2d(self.num_s, num_in, kernel_size=kernel_size, padding=padding, stride=(1,1), 
                              groups=1, bias=False)
        self.blocker = nn.BatchNorm2d(num_in) 

    def forward(self, x):
        batch_size = x.size(0)
        # generate projection and inverse projection matrices
        x_state_reshaped = self.conv_state(x).view(batch_size, self.num_s, -1) 
        x_proj_reshaped = self.conv_proj(x).view(batch_size, self.num_n, -1)
        x_rproj_reshaped = self.conv_reproj(x).view(batch_size, self.num_n, -1)
        # project to node space
        x_n_state1 = torch.bmm(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1)) 
        x_n_state2 = x_n_state1 * (1. / x_state_reshaped.size(2))
        # graph convolution
        x_n_rel1 = self.gcn1(x_n_state2)  
        x_n_rel2 = self.gcn2(x_n_rel1)
        # inverse project to original space
        x_state_reshaped = torch.bmm(x_n_rel2, x_rproj_reshaped)
        x_state = x_state_reshaped.view(batch_size, self.num_s, *x.size()[2:])
        # fusion
        out = x + self.blocker(self.fc_2(x_state))

        return out

def img2df(img, mask):
        img[mask == 0] = 0
        img[mask == 2] = 0
        return img

def feature_fusion(out1, out2):
        output2 = F.log_softmax(out2, dim=1)
        out1_bg = torch.zeros([out1.shape[0], 1, out1.shape[2], out1.shape[3]]).cuda()
        out1_disc = torch.zeros([out1.shape[0], 1, out1.shape[2], out1.shape[3]]).cuda()
        out2_layer = torch.zeros([out2.shape[0], 9, out2.shape[2], out2.shape[3]]).cuda()
        out1_bg[:, 0, :, :] = out1[:, 0, :, :]
        out1_disc[:, 0, :, :] = out1[:, 2, :, :]
        out2_layer[:, :, :, :] = out2[:, 1:, :, :]
        out = torch.cat([out1_bg, out2_layer, out1_disc], 1)
        return output2, out
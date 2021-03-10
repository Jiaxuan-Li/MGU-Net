from models.nets.TSNet import TSMGUNet,TSBaseline,TSMUNet,TSGUNet
from models.nets.OSNet import OSMGUNet,OSBaseline,OSReLayNet,OSUNetbase,OSDRUNet
from models.nets.UNet import UNet
from models.nets.TSNet2 import TSMGUNet2,TSMGUNet2_1,TSMGUNet2_2,TSMGUNet4,TSMGUNet5
from models.nets.TSNet3 import TSMGUNet3,TSMGUNet3_1,TSMGUNet3_2

def net_builder(name,pretrained_model=None,pretrained=False):
    if('resunet50' in name.lower()):
        net = resnet50_UNet(pretrained=pretrained)
    elif name == 'unet':
        net = UNet(n_classes=11,feature_scale=4)    ######################
    elif name == 'tsmgunet':
        net = TSMGUNet()
    elif name == 'tsmgunet2':
        net = TSMGUNet2()
    elif name == 'tsmgunet2_1':
        net = TSMGUNet2_1()
    elif name == 'tsmgunet2_2':
        net = TSMGUNet2_2()
    elif name == 'tsmgunet4':
        net = TSMGUNet4()
    elif name == 'tsmgunet5':
        net = TSMGUNet5()
    elif name == 'tsmgunet3':
        net = TSMGUNet3()
    elif name == 'tsmgunet3_1':
        net = TSMGUNet3_1()
    elif name == 'tsmgunet3_2':
        net = TSMGUNet3_2()
    elif name == 'tsbaseline':
        net = TSBaseline()
    elif name == 'tsmunet':
        net = TSMUNet()
    elif name == 'tsgunet':
        net = TSGUNet()
    elif name == 'osmgunet':
        net = OSMGUNet()
    elif name == 'osbaseline':
        net = OSBaseline()
    elif name == 'osrelaynet':
        net = OSReLayNet()
    elif name == 'osunetbase':
        net = OSUNetbase()
    elif name == 'osdrunet':
        net = OSDRUNet()
    else:
        raise NameError("Unknow Model Name!")
    return net

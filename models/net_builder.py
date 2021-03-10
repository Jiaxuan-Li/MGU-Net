from models.nets.TSNet import TSMGUNet
from models.nets.OSNet import OSMGUNet

def net_builder(name,pretrained_model=None,pretrained=False):
    if name == 'tsmgunet':
        net = TSMGUNet()
    elif name == 'osmgunet':
        net = OSMGUNet()
    else:
        raise NameError("Unknow Model Name!")
    return net

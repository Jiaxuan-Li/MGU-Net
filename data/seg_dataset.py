from torch.utils.data import Dataset
from os.path import join,exists
from PIL import Image
import torch
import os
import os.path as osp
import numpy as np 
import torchvision.transforms as tt
import data.seg_transforms as st
import PIL
import random


class segList(Dataset):
    def __init__(self, data_dir, phase, transforms):
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        if self.phase == 'train':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            data = [Image.open(self.image_list[index])]
            data.append(Image.open(self.label_list[index]))
            data = list(self.transforms(*data))
            data = [data[0],data[1].long()]
            return tuple(data)
        
        if self.phase == 'eval' or 'test':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            self.label_list = get_list_dir(self.phase, 'mask', self.data_dir)
            data = [Image.open(self.image_list[index])]
            imt = torch.from_numpy(np.array(data[0]))
            data.append(Image.open(self.label_list[index]))
            data = list(self.transforms(*data))
            image = data[0]
            label = data[1]
            imn = self.image_list[index].split('/')[-1]
            return (image,label.long(),imt,imn)

        if self.phase == 'predict':
            self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
            data = [Image.open(self.image_list[index])]
            imt = torch.from_numpy(np.array(data[0]))
            data = list(self.transforms(*data))
            image = data[0]
            imn = self.image_list[index].split('/')[-1]
            return (image,imt,imn)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):    
        self.image_list = get_list_dir(self.phase, 'img', self.data_dir)
        print('Total amount of {} images is : {}'.format(self.phase, len(self.image_list)))

def get_list_dir(phase, type, data_dir):
    data_dir = osp.join(data_dir, phase, type)
    files = os.listdir(data_dir)
    list_dir = []
    for file in files:
        file_dir = osp.join(data_dir, file)
        list_dir.append(file_dir)
    return list_dir


    

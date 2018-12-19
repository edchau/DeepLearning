#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 19:18:11 2018

@author: Edward
"""
import torch
from torch.utils.data import Dataset

class toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.x = 2 * torch.ones(length,2)
        self.y = torch.ones(length, 1)
        self.len = length
        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.len

class add_mult(object):
    def __init__(self, addx=1, muly=1):
        self.addx = addx
        self.muly = muly
        
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x,y
        return sample
    
"""
transform
"""
dataset = toy_set()
a_m = add_mult()
x_,y_=a_m(dataset[0])

dataset_ = toy_set(transform=a_m)

"""
transform compose
"""
class mult(object):
    def __init__(self, mul = 100):
        self.mul = mul
    
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mul
        y = y * self.mul
        sample = x,y
        return sample

from torchvision import transforms

data_transform = transforms.Compose([add_mult(), mult()])
data_set_tr = toy_set(transform = data_transform)
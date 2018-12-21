#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 18:33:25 2018

@author: Edward
"""

import torch
from torch.utils.data import Dataset,DataLoader

class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-3,3,0.1).view(-1,1)
        self.y=-3*x+1
        self.len=self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len

dataset = Data()

import torch.nn as nn

class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR,self).__init__()
        self.linear=nn.Linear(in_size, out_size)
    
    def forward(self,x):
        out=self.linear(x)
        return out

#def criterion(yhat, y):
#    return torch.mean((yhat-y)**2)

criterion = nn.MSELoss()
trainloader = DataLoader(dataset=dataset, batch_size=1)

model = LR(1,1)
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for x,y in trainloader:
        yhat = model(x)
        loss = criterion(yhat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
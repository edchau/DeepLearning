#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:27:38 2018

@author: Edward
"""

import torch
import matplotlib.pyplot as plt

"""
Stochastic
"""
w = torch.tensor(-15.0, requires_grad=True)
b = torch.tensor(-10.0, requires_grad=True)

X = torch.arange(-3,3,0.1).view(-1,1)
f = -3*X
plt.plot(X.numpy(), f.numpy())
plt.show()
Y=f+0.1*torch.randn(X.size())

plt.plot(X.numpy(), f.numpy())
plt.plot(X.numpy(), Y.numpy(), 'ro')

def forward(x):
    y=w*x+b
    return y

def criterion(yhat,y):
    return torch.mean((yhat-y)**2)

lr=0.1
LOSS1 = []

for epoch in range(4):
    Yhat=forward(X)
    LOSS1.append(criterion(Yhat,Y))
    for x,y in zip(X,Y):
        yhat=forward(x)
        loss=criterion(yhat,y)
        loss.backward()
        w.data=w.data-lr*w.grad.data
        b.data=b.data-lr*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class Data(Dataset):
    def __init__(self):
        self.x=torch.arange(-3,3,0.1).view(-1,1)
        self.f=-3*X+1
        self.y=self.f+0.1*torch.randn(self.x.size())
        self.len=self.x.shape[0]
    
    def __getitem__(self,index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.len
    
dataset = Data()
#print(dataset.x)
#print(dataset.y)

trainloader=DataLoader(dataset=dataset,batch_size=1) #minibatch -> set different batch_size

for epoch in range(4):
    Yhat=forward(X)
    LOSS1.append(criterion(Yhat,Y))
    for x,y in trainloader:
        yhat=forward(x)
        loss=criterion(yhat,y)
        loss.backward()
        w.data=w.data-lr*w.grad.data
        b.data=b.data-lr*b.grad.data
        w.grad.data.zero_()
        b.grad.data.zero_()

"""
mini-batch
"""
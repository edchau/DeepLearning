#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 21:25:40 2018

@author: Edward
"""

"""
Training Param
"""
import torch
import matplotlib.pyplot as plt 

w = torch.tensor(-10.0, requires_grad=True)
X = torch.arange(-3,3,0.1).view(-1,1) #reshape tensor
"""
the convention in machine learning is each row represents a sample and each 
column represents a feature. As a result, if you have one feature you need to 
specify the extra dimension. 
"""
f = -3*X
plt.plot(X.numpy(),f.numpy())
plt.show

Y = f+0.1*torch.randn(X.size()) #random noise
plt.plot(X.numpy(), f.numpy())
plt.plot(X.numpy(), Y.numpy(), 'ro')

def forward(x):
    y = w * x
    return y

def criterion(yhat, y): #loss
    return torch.mean((yhat-y)**2)

lr = 0.1
LOSS=[]
for epoch in range(3):
    Yhat=forward(X)
    loss=criterion(Yhat,Y)
    loss.backward()
    w.data=w.data-lr*w.grad
    w.grad.data.zero_()
    LOSS.append(loss)
    
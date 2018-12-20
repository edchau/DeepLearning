#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 20:17:59 2018

@author: Edward
"""

"""
Linear Regression
"""

#y = b + wx
#predictor(independent) - x, target(dependent) - y, bias - b, slope - w
#datapoints -> train model -> use model param to make prediction

import torch
w=torch.tensor(2.0, requires_grad = True)
b=torch.tensor(-1.0, requires_grad = True)

def forward(x):
    y = w * x + b
    return y

x = torch.tensor([[1.0]])
yhat = forward(x)

x = torch.tensor([[1.0],[2.0]])
yhat=forward(x) #apply linear equation to every element in tensor

"""
Linear Class
"""
from torch.nn import Linear

torch.manual_seed(1)
model = Linear(in_features=1, out_features=1) #random param
y = model(x) #construct linear model
print(list(model.parameters()))
#yhat = -.44+.51x
x=torch.tensor([[0.]])
yhat = model(x)

x=torch.tensor([[1.0],[2.0]]) #lin eq applied to every row of tensor
yhat=model(x)

"""
Custom Modules
"""
import torch.nn as nn
class LR(nn.Module):
    def __init__(self,in_size,out_size): #determine in/out size
        nn.Module.__init__(self)
        self.linear=nn.Linear(in_size,out_size)
    
    def forward(self, x):
        out = self.linear(x)
        return out
    
model = LR(1,1)
print(list(model.parameters()))
x = torch.tensor([[1.0]])
yhat = model(x)
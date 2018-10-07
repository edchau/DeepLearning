#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 22:58:49 2018

@author: Edward
"""

import torch
import numpy as np
import pandas as pd


"""
Basics
"""
#tensor: matrix containing array of single data type
a = torch.tensor([0,1,2,3,4]) 
a.dtype #64int
a.type() #LongTensor

a = torch.FloatTensor([0,1,2,3,4]) #cast ints to float
a = a.type(torch.FloatTensor)

a.size() #5
a.ndimension() #1

#reshape tensor to column 
a_col = a.view(5,1) 
a_col = a.view(-1,1)  #-1 if unsure of tensor size

#can use numpy array with tensors
torch_tensor = torch.from_numpy(np.array([0.0,1.0,2.0,3.0,4.0]))
convert_back = torch_tensor.numpy()

#can usewith pandas
pandas_series = pd.Series([0.1,2,0.3,10.1])
pandas_to_torch = torch.from_numpy(pandas_series.values)

"""
Indexing and Slicing
"""
c=torch.tensor([20,1,2,3,4])
c[0]=100
c[4]=0
d=c[1:4] #[1, 2, 3]
c[3:5]=torch.tensor([300,400])


"""
Basic Operations
"""
#tensors behave like vectors
u = torch.tensor([1,0])
v = torch.tensor([0,1])
z=u+v

y = torch.tensor([1,2])
z = 2*y #scalar product

u = torch.tensor([1,2])
v = torch.tensor([3,2])
z = u*v #[3,4]

#dot product
result = torch.dot(u,v)

u = torch.tensor([1,2,3,-1])
z  = u+1 #[2,3,4,0]

"""
Universal Functions
"""
a = torch.tensor([1.0,-1,1,-1])
mean_a = a.mean()
b = np.array([1,-2,3,4,5])
max_b = b.max

np.pi
x = torch.tensor([0,np.pi/2,np.pi])
y = torch.sin(x) #assigns every element in tensor to y

torch.linspace(-2,2,steps=5) #-2,-1,0,1,2
x = torch.linspace(0,2*np.pi,100)
y = torch.sin(x)

import matplotlib.pyplot as plt
plt.plot(x.numpy(), y.numpy())
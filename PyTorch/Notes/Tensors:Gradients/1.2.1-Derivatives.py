#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:06:30 2018

@author: Edward
"""

import torch
import matplotlib.pyplot as plt

"""
Derivatives
"""

x = torch.tensor(2.0, requires_grad=True)
#requires_grad stores all the operations associated with the variable
#https://pytorch.org/docs/stable/notes/autograd.html

y=x**2
y.backward()
#derivative of y with respect to x at 2

print(x.grad)
#value of deriv of y with respect to x at 2

z = x**2+2*x+1
z.backward()
print(x.grad)

#x.grad.zero_()
#https://stackoverflow.com/questions/48001598/why-is-zero-grad-needed-for-optimization
#set gradients to zero before backpropagation

#z.backward(retain_graph=True)


"""
Partial Derivatives
"""
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f=u*v+u**2
f.backward()
print(u.grad)
print(v.grad)


#derivative of y=x^2 w.r.t. values from -10 to 10
x=torch.linspace(-10,10,10,requires_grad=True)
Y=x**2
y=torch.sum(x**2)
y.backward()

#when we convert a tensor with the parameter requires grad equal to
#true, we have to use the detach .option before we can cast it as this numpy array.

plt.plot(x.detach().numpy(), Y.detach().numpy(), label='function')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='derivative')
plt.legend()





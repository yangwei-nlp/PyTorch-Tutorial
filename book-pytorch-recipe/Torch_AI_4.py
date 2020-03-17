#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


# In[2]:


#torch.nn: - Neural networks can be constructed using the torch.nn package.


# In[3]:


x = Variable(torch.randn(100, 10))
y = Variable(torch.randn(100, 30))

linear = nn.Linear(in_features=10, out_features=5, bias=True)
output_linear = linear(x)
print('Output size : ', output_linear.size())

bilinear = nn.Bilinear(in1_features=10, in2_features=30, out_features=5, bias=True)
output_bilinear = bilinear(x, y)
print('Output size : ', output_bilinear.size())


# In[4]:


x = Variable(torch.randn(100, 10))
y = Variable(torch.randn(100, 30))

sig = nn.Sigmoid()
output_sig = sig(x)
output_sigy = sig(y)
print('Output size : ', output_sig.size())
print('Output size : ', output_sigy.size())


# In[5]:


print(x[0])
print(output_sig[0])


# In[6]:


x = Variable(torch.randn(100, 10))
y = Variable(torch.randn(100, 30))

func = nn.Tanh()
output_x = func(x)
output_y = func(y)
print('Output size : ', output_x.size())
print('Output size : ', output_y.size())


# In[7]:


print(x[0])
print(output_x[0])
print(y[0])
print(output_y[0])


# In[8]:


x = Variable(torch.randn(100, 10))
y = Variable(torch.randn(100, 30))

func = nn.LogSigmoid()
output_x = func(x)
output_y = func(y)
print('Output size : ', output_x.size())
print('Output size : ', output_y.size())


# In[9]:


print(x[0])
print(output_x[0])
print(y[0])
print(output_y[0])


# In[10]:


x = Variable(torch.randn(100, 10))
y = Variable(torch.randn(100, 30))

func = nn.ReLU()
output_x = func(x)
output_y = func(y)
print('Output size : ', output_x.size())
print('Output size : ', output_y.size())


# In[11]:


print(x[0])
print(output_x[0])
print(y[0])
print(output_y[0])


# In[12]:


x = Variable(torch.randn(100, 10))
y = Variable(torch.randn(100, 30))

func = nn.LeakyReLU()
output_x = func(x)
output_y = func(y)
print('Output size : ', output_x.size())
print('Output size : ', output_y.size())


# In[13]:


print(x[0])
print(output_x[0])
print(y[0])
print(output_y[0])


# In[31]:


import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


x = torch.linspace(-10, 10, 1500)  
x = Variable(x)
x_1 = x.data.numpy()   # tranforming into numpy


# In[33]:


y_relu = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()


# In[34]:


plt.figure(figsize=(7, 4))
plt.plot(x_1, y_relu, c='blue', label='ReLU')
plt.ylim((-1, 11))
plt.legend(loc='best')


# In[35]:


plt.figure(figsize=(7, 4))
plt.plot(x_1, y_sigmoid, c='blue', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')


# In[36]:


plt.figure(figsize=(7, 4))
plt.plot(x_1, y_tanh, c='blue', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')


# In[37]:


plt.figure(figsize=(7, 4))
plt.plot(x_1, y_softplus, c='blue', label='softplus')
plt.ylim((-0.2, 11))
plt.legend(loc='best')


# In[185]:


def prep_data():
    train_X = np.asarray([13.3,14.4,15.5,16.71,16.93,14.168,19.779,16.182,
                          17.59,12.167,17.042,10.791,15.313,17.997,15.654,
                          19.27,13.1])
    train_Y = np.asarray([11.7,12.76,12.09,13.19,11.694,11.573,13.366,12.596,
                          12.53,11.221,12.827,13.465,11.65,12.904,12.42,12.94,
                          11.3])
    dtype = torch.FloatTensor
    X = Variable(torch.from_numpy(train_X).type(dtype),
                 requires_grad=False).view(17,1)
    y = Variable(torch.from_numpy(train_Y).type(dtype),requires_grad=False)
    return X,y


# In[186]:


# get dynamic parameters


# In[187]:


def set_weights():
    w = Variable(torch.randn(1),requires_grad = True)
    b = Variable(torch.randn(1),requires_grad=True)
    return w,b


# In[188]:


#deploy neural network model


# In[189]:


def build_network(x):
    y_pred = torch.matmul(x,w)+b
    return y_pred


# In[190]:


#implement in PyTorch
import torch.nn as nn
f = nn.Linear(17,1) # Much simpler.
f


# In[191]:


#calculate the loss function


# In[210]:


def loss_calc(y,y_pred):
    loss = (y_pred-y).pow(2).sum()
    for param in [w,b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()
    return loss.data[0]


# In[211]:


# optimizing results


# In[212]:


def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


# In[208]:


learning_rate = 1e-4


# In[216]:


x,y = prep_data()  # x - training data,y - target variables
w,b = set_weights() # w,b - parameters
for i in range(5000):
    y_pred = build_network(x) # function which computes wx + b
    loss = loss_calc(y,y_pred) # error calculation
    if i % 1000 == 0: 
        print(loss)
    optimize(learning_rate)    # minimize the loss w.r.t. w, b


# In[197]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[198]:


x_numpy = x.data.numpy()
y_numpy = y.data.numpy()
y_pred = y_pred.data.numpy()
plt.plot(x_numpy,y_numpy,'o')
plt.plot(x_numpy,y_pred,'-')


# In[219]:


x = Variable(torch.ones(4, 4) * 12.5, requires_grad=True)


# In[220]:


x


# In[221]:


fn = 2 * (x * x) + 5 * x + 6

# 2x^2 + 5x + 6


# In[224]:


fn.backward(torch.ones(4,4))


# In[225]:


print(x.grad)


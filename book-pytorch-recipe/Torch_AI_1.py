#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import torch


# In[2]:


torch.version.__version__


# # PyTorch basics

# The torch package contains data structures for multi-dimensional tensors and mathematical operations over these are
# defined. Additionally, it provides many utilities for efficient serializing of Tensors and arbitrary types, and other useful
# utilities.
# It has a CUDA counterpart, that enables you to run your tensor computations on an NVIDIA GPU with compute
# capability >= 2.0.

# In[6]:


x = [12,23,34,45,56,67,78]


# In[7]:


torch.is_tensor(x)


# In[8]:


torch.is_storage(x)


# In[9]:


y = torch.randn(1,2,3,4,5)


# In[10]:


torch.is_tensor(y)


# In[11]:


torch.is_storage(y)


# In[12]:


torch.numel(y) # the total number of elements in the input Tensor


# In[13]:


torch.zeros(4,4)


# In[14]:


torch.numel(torch.zeros(4,4))


# In[15]:


torch.eye(3)


# In[16]:


torch.eye(5)


# In[17]:


torch.eye(3,4)


# In[18]:


torch.eye(5,4)


# In[19]:


type(x)


# In[20]:


import numpy as np
x1 = np.array(x)


# In[21]:


x1


# In[22]:


torch.from_numpy(x1)


# In[20]:


torch.linspace(2, 10, steps=25) #linear spacing


# In[23]:


torch.linspace(-10, 10, steps=15)


# In[24]:


torch.logspace(start=-10, end=10, steps=15) #logarithmic spacing


# In[25]:


torch.ones(4)


# In[26]:


torch.ones(4,5)


# In[27]:


# random numbers from a uniform distribution between the values 
# 0 and 1
torch.rand(10)


# In[28]:


torch.rand(4, 5) 
# random values between 0 and 1 and fillied with a matrix of 
# size rows 4 and columns 5


# In[29]:


#random numbers from a normal distribution, 
#with mean =0 and standard deviation =1
torch.randn(10)


# In[30]:


torch.randn(4, 5)


# In[31]:


#selecting values from a range, this is called random permutation
torch.randperm(10)


# In[32]:


#usage of range function 
torch.arange(10, 40,2) #step size 2


# In[33]:


torch.arange(10,40) #step size 1


# In[34]:


d = torch.randn(4, 5)
d


# In[35]:


torch.argmin(d,dim=1)


# In[36]:


torch.argmax(d,dim=1)


# In[37]:


# create a 2dtensor filled with values as 0
torch.zeros(4,5)


# In[38]:


# create a 1d tensor filled with values as 0
torch.zeros(10)


# In[39]:


#indexing and performing operation on the tensors
x = torch.randn(4,5)


# In[40]:


x


# In[41]:


#concatenate two tensors
torch.cat((x,x))


# In[42]:


#concatenate n times based on array size
torch.cat((x,x,x))


# In[43]:


#concatenate n times based on array size, over column
torch.cat((x,x,x),1)


# In[44]:


#concatenate n times based on array size, over rows
torch.cat((x,x),0)


# In[45]:


#how to split a tensor among small chunks


# In[46]:


help(torch.chunk)


# In[47]:


a = torch.randn(4, 4)
print(a)
torch.chunk(a,2)


# In[48]:


torch.chunk(a,2,0)


# In[49]:


torch.chunk(a,2,1)


# In[50]:


torch.Tensor([[11,12],[23,24]])


# In[51]:


torch.gather(torch.Tensor([[11,12],[23,24]]), 1, 
             torch.LongTensor([[0,0],[1,0]]))


# In[52]:


torch.LongTensor([[0,0],[1,0]]) 
#the 1D tensor containing the indices to index


# In[53]:


a = torch.randn(4, 4)
print(a)


# In[54]:


indices = torch.LongTensor([0, 2])


# In[55]:


torch.index_select(a, 0, indices)


# In[56]:


torch.index_select(a, 1, indices)


# In[57]:


#identify null input tensors using nonzero function
torch.nonzero(torch.tensor([10,00,23,0,0.0]))


# In[58]:


torch.nonzero(torch.Tensor([10,00,23,0,0.0]))


# In[59]:


# splitting the tensor into small chunks
torch.split(torch.tensor([12,21,34,32,45,54,56,65]),2)


# In[58]:


# splitting the tensor into small chunks
torch.split(torch.tensor([12,21,34,32,45,54,56,65]),3)


# In[59]:


torch.zeros(3,2,4)


# In[60]:


torch.zeros(3,2,4).size()


# In[61]:


#how to reshape the tensors along a new dimension


# In[62]:


x


# In[63]:


x.t() #transpose is one option to change the shape of the tensor


# In[64]:


# transpose partially based on rows and columns


# In[65]:


x.transpose(1,0)


# In[66]:


# how to remove a dimension from a tensor


# In[67]:


x


# In[68]:


torch.unbind(x,1) #dim=1 removing a column


# In[69]:


torch.unbind(x) #dim=0 removing a row


# In[70]:


x


# In[71]:


#how to compute the basic mathematrical functions


# In[72]:


torch.abs(torch.FloatTensor([-10, -23, 3.000]))


# In[73]:


#adding value to the existing tensor, scalar addition
torch.add(x,20)


# In[74]:


x


# In[75]:


# scalar multiplication
torch.mul(x,2)


# In[76]:


x


# In[77]:


# how do we represent the equation in the form of a tensor


# In[78]:


# y = intercept + (beta * x)


# In[79]:


intercept = torch.randn(1)
intercept


# In[80]:


x = torch.randn(2, 2)
x


# In[81]:


beta = 0.7456
beta


# In[82]:


torch.mul(x,beta)


# In[83]:


torch.add(x,beta,intercept)


# In[84]:


torch.mul(intercept,x)


# In[85]:


torch.mul(x,beta)


# In[86]:


## y = intercept + (beta * x)
torch.add(torch.mul(intercept,x),torch.mul(x,beta)) # tensor y


# In[87]:


# how to round up tensor values
torch.manual_seed(1234)
torch.randn(5,5)


# In[88]:


torch.manual_seed(1234)
torch.ceil(torch.randn(5,5))


# In[89]:


torch.manual_seed(1234)
torch.floor(torch.randn(5,5))


# In[90]:


# truncate the values in a range say 0,1
torch.manual_seed(1234)
torch.clamp(torch.floor(torch.randn(5,5)), min=-0.3, max=0.4)


# In[91]:


#truncate with only lower limit
torch.manual_seed(1234)
torch.clamp(torch.floor(torch.randn(5,5)), min=-0.3)


# In[92]:


#truncate with only upper limit
torch.manual_seed(1234)
torch.clamp(torch.floor(torch.randn(5,5)), max=0.3)


# In[93]:


#scalar division
torch.div(x,0.10)


# In[94]:


#compute the exponential of a tensor
torch.exp(x)


# In[95]:


np.exp(x)


# In[96]:


#how to get the fractional portion of each tensor


# In[97]:


torch.add(x,10)


# In[98]:


torch.frac(torch.add(x,10))


# In[99]:


# compute the log of the values in a tensor


# In[100]:


x


# In[101]:


torch.log(x) #log of negatives are nan


# In[102]:


# to rectify the negative values do a power tranforamtion
torch.pow(x,2)


# In[103]:


# rounding up similar to numpy


# In[104]:


x


# In[105]:


np.round(x)


# In[106]:


torch.round(x)


# In[107]:


# how to compute the sigmoid of the input tensor


# In[108]:


x


# In[109]:


torch.sigmoid(x)


# In[110]:


# finding the square root of the values


# In[111]:


x


# In[112]:


torch.sqrt(x)


# In[ ]:





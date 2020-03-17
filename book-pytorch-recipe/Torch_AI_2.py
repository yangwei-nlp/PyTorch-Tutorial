#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


# how to perform random sampling of the tensors


# In[3]:


torch.manual_seed(1234)


# In[4]:


torch.manual_seed(1234)
torch.randn(4,4)


# In[5]:


#generate random numbers from a statistical distribution


# In[6]:


torch.Tensor(4, 4).uniform_(0, 1) #random number from uniform distribution


# In[7]:


#now apply the distribution assuming the input values from the 
#tensor are probabilities


# In[8]:


torch.bernoulli(torch.Tensor(4, 4).uniform_(0, 1))


# In[9]:


#how to perform sampling from a multinomial distribution


# In[10]:


torch.Tensor([10, 10, 13, 10,34,45,65,67,87,89,87,34])


# In[11]:


torch.multinomial(torch.tensor([10., 10., 13., 10., 
                                34., 45., 65., 67., 
                                87., 89., 87., 34.]), 
                  3)


# In[12]:


torch.multinomial(torch.tensor([10., 10., 13., 10., 
                                34., 45., 65., 67., 
                                87., 89., 87., 34.]), 
                  5, replacement=True)


# In[13]:


#generate random numbers from the normal distribution


# In[14]:


torch.normal(mean=torch.arange(1., 11.), 
             std=torch.arange(1, 0, -0.1))


# In[15]:


torch.normal(mean=0.5, 
             std=torch.arange(1., 6.))


# In[16]:


torch.normal(mean=0.5, 
             std=torch.arange(0.2,0.6))


# In[17]:


#computing the descriptive statistics: mean
torch.mean(torch.tensor([10., 10., 13., 10., 34., 
                         45., 65., 67., 87., 89., 87., 34.]))


# In[18]:


# mean across rows and across columns
d = torch.randn(4, 5)
d


# In[19]:


torch.mean(d,dim=0)


# In[20]:


torch.mean(d,dim=1)


# In[21]:


#compute median
torch.median(d,dim=0)


# In[22]:


torch.median(d,dim=1)


# In[23]:


# compute the mode
torch.mode(d)


# In[24]:


torch.mode(d,dim=0)


# In[25]:


torch.mode(d,dim=1)


# In[26]:


#compute the standard deviation
torch.std(d)


# In[27]:


torch.std(d,dim=0)


# In[28]:


torch.std(d,dim=1)


# In[29]:


#compute variance
torch.var(d)


# In[30]:


torch.var(d,dim=0)


# In[31]:


torch.var(d,dim=1)


# In[32]:


# compute min and max
torch.min(d)


# In[33]:


torch.min(d,dim=0)


# In[34]:


torch.min(d,dim=1)


# In[35]:


torch.max(d)


# In[36]:


torch.max(d,dim=0)


# In[37]:


torch.max(d,dim=1)


# In[38]:


# sorting a tensor
torch.sort(d)


# In[39]:


torch.sort(d,dim=0)


# In[40]:


torch.sort(d,dim=0,descending=True)


# In[41]:


torch.sort(d,dim=1,descending=True)


# In[42]:


from torch.autograd import Variable


# In[43]:


Variable(torch.ones(2,2),requires_grad=True)


# In[44]:


a, b = 12,23
x1 = Variable(torch.randn(a,b),
            requires_grad=True)
x2 = Variable(torch.randn(a,b),
            requires_grad=True)
x3 =Variable(torch.randn(a,b),
            requires_grad=True)


# In[45]:


c = x1 * x2
d = a + x3
e = torch.sum(d)

e.backward()

print(e)


# In[46]:


x1.data


# In[47]:


x2.data


# In[48]:


x3.data


# In[102]:


from torch import FloatTensor
from torch.autograd import Variable

a = Variable(FloatTensor([5]))

weights = [Variable(FloatTensor([i]), requires_grad=True) for i in (12, 53, 91, 73)]

w1, w2, w3, w4 = weights

b = w1 * a
c = w2 * a
d = w3 * b + w4 * c
Loss = (10 - d)

Loss.backward()

for index, weight in enumerate(weights, start=1):
    gradient, *_ = weight.grad.data
    print(f"Gradient of w{index} w.r.t to Loss: {gradient}")


# In[ ]:





# In[50]:


# Using forward pass
def forward(x):
    return x * w


# In[51]:


import torch
from torch.autograd import Variable

x_data = [11.0, 22.0, 33.0]
y_data = [21.0, 14.0, 64.0]

w = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value

# Before training
print("predict (before training)",  4, forward(4).data[0])


# In[52]:


# define the Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# In[53]:


# Run the Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually set the gradients to zero after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])


# In[54]:


# After training
print("predict (after training)",  4, forward(4).data[0])


# In[55]:


z = Variable(torch.Tensor(4, 4).uniform_(-5, 5))
print(z)


# In[56]:


print('Requires Gradient : %s ' % (z.requires_grad))
print('Volatile : %s ' % (z.volatile))
print('Gradient : %s ' % (z.grad))
print(z.data)


# In[83]:


x = Variable(torch.Tensor(4, 4).uniform_(-4, 5))
y = Variable(torch.Tensor(4, 4).uniform_(-3, 2))
# matrix multiplication
z = torch.mm(x, y)
print(z.size())


# In[84]:


x.data


# In[103]:


#tensor operations


# In[109]:


mat1 = torch.FloatTensor(4,4).uniform_(0,1)
mat1


# In[110]:


mat2 = torch.FloatTensor(5,4).uniform_(0,1)
mat2


# In[111]:


vec1 = torch.FloatTensor(4).uniform_(0,1)
vec1


# In[112]:


# scalar addition


# In[113]:


mat1 + 10.5


# In[114]:


# scalar subtraction


# In[115]:


mat2 - 0.20


# In[116]:


# vector and matrix addition


# In[117]:


mat1 + vec1


# In[118]:


mat2 + vec1


# In[119]:


# matrix-matrix addition


# In[123]:


mat1 + mat2


# In[131]:


mat1 * mat1


# In[ ]:


# about Bernoulli distribution


# In[138]:


from torch.distributions.bernoulli import Bernoulli


# In[139]:


dist = Bernoulli(torch.tensor([0.3,0.6,0.9]))


# In[140]:


dist.sample() #sample is binary, it takes 1 with p and 0 with 1-p


# In[132]:


#Creates a Bernoulli distribution parameterized by probs 
#Samples are binary (0 or 1). They take the value 1 with probability p 
#and 0 with probability 1 - p.


# In[133]:


from torch.distributions.beta import Beta


# In[141]:


dist = Beta(torch.tensor([0.5]), torch.tensor([0.5]))
dist


# In[142]:


dist.sample()


# In[143]:


from torch.distributions.binomial import Binomial


# In[144]:


dist = Binomial(100, torch.tensor([0 , .2, .8, 1]))


# In[147]:


dist.sample()


# In[148]:


# 100- count of trials
# 0, 0.2, 0.8 and 1 are event probabilities


# In[174]:


from torch.distributions.categorical import Categorical


# In[175]:


dist = Categorical(torch.tensor([ 0.20, 0.20, 0.20, 0.20, 0.20 ]))
dist


# In[176]:


dist.sample()


# In[177]:


# 0.20, 0.20, 0.20, 0.20,0.20 event probabilities


# In[155]:


# Laplace distribution parameterized by loc and ‘scale’.


# In[157]:


from torch.distributions.laplace import Laplace


# In[161]:


dist = Laplace(torch.tensor([10.0]), torch.tensor([0.990]))
dist


# In[162]:


dist.sample()


# In[163]:


#Normal (Gaussian) distribution parameterized by loc and ‘scale’.


# In[165]:


from torch.distributions.normal import Normal


# In[166]:


dist = Normal(torch.tensor([100.0]), torch.tensor([10.0]))
dist


# In[167]:


dist.sample()


# In[ ]:





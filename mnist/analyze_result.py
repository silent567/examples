#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


def gather_record(key=''):
    filenames = os.listdir()
    record = []
    for fn in filenames:
        if '.csv' in fn and key in fn:
            tmp_record = np.loadtxt(fn,delimiter=',')
            tmp_record = tmp_record[tmp_record[:,-1]>0]
            record.append(tmp_record)
    record = np.concatenate(record,axis=0)
    return record
record = gather_record('record2_2018')
print(record.shape)
print(record[np.argmax(record[:,-1])])


# In[3]:


index2name = ['lr','norm_flag','gamma','lam','max_type','optim_type']
name2index = {n:i for i,n in enumerate(index2name)}
print(index2name)
print(name2index)


# In[4]:


tmp_record = record[record[:,4]==2]
print(tmp_record[np.argmax(tmp_record[:,-1])])


# In[6]:


#filtered out low accuracies
frecord = record[record[:,-1]>=0.95]
#box plot
index = name2index['max_type']
values = list(set(frecord[:,index]))
values.sort()
plt.boxplot([frecord[frecord[:,index]==v][:,-1] for v in values],labels=['softmax','sparsemax','gfusedmax'])


# In[7]:


#filtered out low accuracies
frecord = record[record[:,-1]>=0.95]
#filtered out non gfusedmax
frecord = frecord[frecord[:,name2index['max_type']]==2]
print(frecord.shape)
#box plot
index = name2index['optim_type']
values = list(set(frecord[:,index]))
values.sort()
plt.boxplot([frecord[frecord[:,index]==v][:,-1] for v in values],labels=['SGD','Adam'])


# In[8]:


#filtered out low accuracies
frecord = record[record[:,-1]>=0.95]
#filtered out non gfusedmax
frecord = frecord[frecord[:,name2index['max_type']]==2]
print(frecord.shape)
#box plot
index = name2index['norm_flag']
values = list(set(frecord[:,index]))
values.sort()
plt.boxplot([frecord[frecord[:,index]==v][:,-1] for v in values],labels=['no-norm','with-norm'])


# In[10]:


#filtered out low accuracies
frecord = record[record[:,-1]>=0.95]
#filtered out non gfusedmax
frecord = frecord[frecord[:,name2index['max_type']]==2]
#filtered out non layer-norm
#frecord = frecord[frecord[:,name2index['norm_flag']]==1]
print(frecord.shape)
#box plot
index = name2index['lam']
values = list(set(frecord[:,index]))
values.sort()
plt.boxplot([frecord[frecord[:,index]==v][:,-1] for v in values],labels=['%.2f'%v for v in values])
plt.xlabel('$\lambda$')


# In[12]:


#filtered out low accuracies
frecord = record[record[:,-1]>=0.95]
#filtered out non gfusedmax
frecord = frecord[frecord[:,name2index['max_type']]==2]
#filtered out non layer-norm
#frecord = frecord[frecord[:,name2index['norm_flag']]==1]
print(frecord.shape,frecord[np.argmax(frecord[:,-1])])
#box plot
index = name2index['gamma']
values = list(set(frecord[:,index]))
values.sort()
plt.boxplot([frecord[frecord[:,index]==v][:,-1] for v in values],labels=['%.2f'%v for v in values])
plt.xlabel('$\gamma$')


# In[13]:


#filtered out low accuracies
frecord = record[record[:,-1]>=0.95]
#filtered out non gfusedmax
frecord = frecord[frecord[:,name2index['max_type']]==2]
#filtered out non layer-norm
frecord = frecord[frecord[:,name2index['norm_flag']]==1]
print(frecord.shape,frecord[np.argmax(frecord[:,-1])])
#box plot
index = name2index['optim_type']
values = list(set(frecord[:,index]))
values.sort()
plt.boxplot([frecord[frecord[:,index]==v][:,-1] for v in values],labels=['SGD','Adam'])


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install kagglehub install kaggle library for downloading training data


# In[6]:


import pandas as pd


# In[3]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

print("Path to dataset files:", path)

#multiclass classification
#The class ids are numbered 1-4 where 1 represents World, 2 represents Sports, 3 represents Business and 4 represents Sci/Tech.


# In[7]:


ag_dataset_test = pd.read_csv(r"/Users/robertheld/.cache/kagglehub/datasets/amananandrai/ag-news-classification-dataset/versions/2/test.csv")
ag_dataset_train = pd.read_csv(r"/Users/robertheld/.cache/kagglehub/datasets/amananandrai/ag-news-classification-dataset/versions/2/train.csv")


# In[8]:


ag_dataset_test


# In[9]:


ag_dataset_train


# In[ ]:





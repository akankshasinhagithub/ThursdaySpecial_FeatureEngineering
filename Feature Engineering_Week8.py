#!/usr/bin/env python
# coding: utf-8

# In[1]:


#DS360withAkanksha | Thursday Special:Feature Engineering


# In[2]:


# Importing required packages

import pandas as pd #data analysis and manipulation tool
import numpy as np #Fundamental package for scientific computing
import seaborn as sns #Data visualization library based on matplotlib.
import tqdm #Instantly make your loops show a smart progress meter
import gc #Provides an interface to the optional garbage collector


# In[3]:


#Defining number of rows 

num_rows = 1000


# In[15]:


#Getting data

all_df = pd.read_csv('application_train.csv', nrows=num_rows)
test_df = pd.read_csv('application_test.csv', nrows=num_rows)
all_df = all_df.append(test_df).reset_index()


# In[16]:


#Top rows

all_df.head()


# In[17]:


#Descriptive Statistics
all_df.describe()


# In[18]:


all_df.info()


# In[19]:


# Getting rid of outliers

all_df = all_df[all_df['CODE_GENDER'] != 'XNA']
all_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)


# In[20]:


# Getting all repetitive columns with list comprehension

docs = [_f for _f in all_df.columns if 'FLAG_DOC' in _f]
live = [_f for _f in all_df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]

print(docs[:3])
print(live[:3])


# In[24]:


# Median of incomes of different organization

inc_by_org = all_df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']


# In[26]:


#Taking the ratio of amount credit by annuity as we well as amount credit by goods price

all_df['NEW_CREDIT_TO_ANNUITY_RATIO'] = all_df['AMT_CREDIT'] / all_df['AMT_ANNUITY']

all_df['NEW_CREDIT_TO_GOODS_RATIO'] = all_df['AMT_CREDIT'] / all_df['AMT_GOODS_PRICE']


# In[27]:


# Creating new features on previously created feature using statistics

all_df['NEW_DOC_AVG'] = all_df[docs].mean(axis=1)
all_df['NEW_DOC_STD'] = all_df[docs].mean(axis=1)
all_df['NEW_DOC_KURT'] = all_df[docs].mean(axis=1)
all_df['NEW_DOC_SUM'] = all_df[docs].mean(axis=1)
all_df['NEW_DOC_STD'] = all_df[docs].mean(axis=1)
all_df['NEW_DOC_KURT'] = all_df[docs].mean(axis=1)


# In[28]:


# Creating some more features

all_df['NEW_INC_PER_CHLD'] = all_df['AMT_INCOME_TOTAL'] / (1 + all_df['CNT_CHILDREN'])
all_df['NEW_INC_BY_ORG'] = all_df['ORGANIZATION_TYPE'].map(inc_by_org)
all_df['NEW_EMPLOY_TO_BIRTH_RATIO'] = all_df['DAYS_EMPLOYED'] / all_df['DAYS_BIRTH']
all_df['NEW_ANNUITY_TO_INCOME_RATIO'] = all_df['AMT_ANNUITY'] / (1 + all_df['AMT_INCOME_TOTAL'])
all_df['NEW_SOURCES_PROD'] = all_df['EXT_SOURCE_1'] * all_df['EXT_SOURCE_2'] * all_df['EXT_SOURCE_3']
all_df['NEW_EXT_SOURCES_MEAN'] = all_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
all_df['NEW_SCORES_STD'] = all_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
all_df['NEW_SCORES_STD'] = all_df['NEW_SCORES_STD'].fillna(all_df['NEW_SCORES_STD'].mean())
all_df['NEW_CAR_TO_BIRTH_RATIO'] = all_df['OWN_CAR_AGE'] / all_df['DAYS_BIRTH']
all_df['NEW_CAR_TO_EMPLOY_RATIO'] = all_df['OWN_CAR_AGE'] / all_df['DAYS_EMPLOYED']
all_df['NEW_PHONE_TO_BIRTH_RATIO'] = all_df['DAYS_LAST_PHONE_CHANGE'] / all_df['DAYS_BIRTH']
all_df['NEW_PHONE_TO_EMPLOY_RATIO'] = all_df['DAYS_LAST_PHONE_CHANGE'] / all_df['DAYS_EMPLOYED']
all_df['NEW_CREDIT_TO_INCOME_RATIO'] = all_df['AMT_CREDIT'] / all_df['AMT_INCOME_TOTAL']


# In[ ]:





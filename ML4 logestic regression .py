#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


df=pd.read_csv('insurance_data.csv')## 
df


# In[24]:


plt.scatter(df.age,df.bought_insurance,marker='+',color='red')


# In[43]:


X=df[['age']]
y=df[['bought_insurance']]

from sklearn.model_selection import train_test_split


# In[33]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.1)


# In[34]:


len(X_train)


# In[35]:


len(X_test)


# In[38]:


from sklearn.linear_model import LogisticRegression
df1=LogisticRegression


# In[39]:


df1.fit(X_train,y_train)## dont know why error is comeing 


# In[48]:


lin_reg=linear_model.LogisticRegression()
lin_reg.fit(df[['age']],df.bought_insurance)


# In[49]:


lin_reg=linear_model.LogisticRegression()## this method is also correct for logistic regression 
lin_reg.fit(X_train,y_train)


# In[50]:


lin_reg.predict(X_test)


# In[51]:


y_test## here 1 mean yes and 0 mean no 


# In[52]:


X_test


# In[53]:


lin_reg.score(X_test,y_test)


# In[56]:


lin_reg.predict_proba(X_test)


# In[59]:


lin_reg=linear_model.LogisticRegression()
lin_reg.fit(df[['age']],df.bought_insurance)


# In[62]:


lin_reg.predict([[50]])## yes 


# In[63]:


lin_reg.predict([[20]])## no 


# In[ ]:





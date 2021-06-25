#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


df=pd.read_csv('carprices2.csv')
df


# In[41]:


plt.scatter(df['Mileage'],df['Sell Price($)'],color='r',marker='+')


# In[42]:


plt.scatter(df['Age(yrs)'],df['Sell Price($)'],color='g',marker='o')


# In[96]:


X = df[['Mileage','Age(yrs)']]
y = df['Sell Price($)']


# In[97]:


X


# In[98]:


y


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)


# In[101]:


len(X_train)


# In[102]:


len(X_test)


# In[103]:


X_train


# In[104]:


from sklearn.linear_model import LinearRegression
df1=LinearRegression()


# In[105]:


df1.fit(X_train,y_train)


# In[106]:


df1.predict(X_test)


# In[107]:


y_test


# In[108]:


df1.score(X_test,y_test)## 92 % accuracy 


# In[ ]:





# In[ ]:





# In[ ]:





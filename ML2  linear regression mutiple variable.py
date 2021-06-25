#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


df=pd.read_csv('homeprices.csv')
df


# In[24]:


## here price is dependent on three factors area,bedrooms,age so we can say that price=m1*area+m2*age+m3*bedrooms+constant   and the equation can be genrealized as y=m1*x1+M2*X2+m3x3+constant


# In[29]:


p=df.bedrooms.median()## findding out medium 
p


# In[53]:


df.bedrooms=df.bedrooms.fillna(p)## filling the missing value df.bedrrroms means we are filling the missing value in bedrooms
df


# In[31]:


import math ## another way to find out the medium value
any_value = math.floor(df.bedrooms.median())
any_value


# In[34]:


df## data preprocessing has done lets predict the prices 


# In[36]:


lin_reg=linear_model.LinearRegression()
lin_reg.fit(df[['area','bedrooms','age']],df.price)


# In[38]:


lin_reg.predict([[3400,4,20]])


# In[41]:


lin_reg.predict([[4500,7,15]])


# In[43]:


lin_reg.coef_ ## cofficient m1,m2,m3


# In[45]:


lin_reg.intercept_


# In[46]:


lin_reg.predict([[3000,3,40]])##Q1


# In[48]:


lin_reg.predict([[2500,4,5]])##q2


# In[52]:


plt.xlabel=['area']
plt.ylabel=['price']
plt.title=['homeprices']
plt.scatter(df.area,df.price,color='r',marker='+')


# In[ ]:





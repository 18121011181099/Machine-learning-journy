#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('carprices.csv')
df


# In[23]:


df=df.rename(columns={'Car Model': 'car_model','Sell Price($)':'sell_price'})
df


# In[24]:


df_dummies=pd.get_dummies(df.car_model)
df_dummies


# In[25]:


merged=pd.concat([df,df_dummies],axis='columns')
merged


# In[33]:


final=merged.drop(['BMW X5','car_model'],axis='columns')
final=final.rename(columns={'Audi A5': 'audi_a5','Mercedez Benz C class':'mercedezBenz_Cclass'})

final


# In[36]:


lin_reg=linear_model.LinearRegression()
lin_reg.fit(final[['Mileage','Age(yrs)','audi_a5','mercedezBenz_Cclass']],df.sell_price)


# In[37]:


lin_reg.predict([[45000,4,0,1]])


# In[39]:


lin_reg.predict([[86000,7,0,0]])


# In[40]:


### how to calculate accuracy of model


# In[41]:


from sklearn.linear_model import LinearRegression
model=LinearRegression


# In[42]:


X=final.drop('sell_price',axis='columns') ## training dataset
X


# In[52]:


y=final['sell_price']
y


# In[53]:


model.fit(X,y)## dont know why its is occuring


# In[ ]:





# In[ ]:





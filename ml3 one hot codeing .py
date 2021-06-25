#!/usr/bin/env python
# coding: utf-8

# In[67]:


## method of createing dummy variables 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


df=pd.read_csv('homeprices12.csv')
df


# In[69]:


df_dummies=pd.get_dummies(df.town)
df_dummies


# In[70]:


merged=pd.concat([df,df_dummies],axis='columns')
merged


# In[71]:


final=merged.drop(['town','west windsor'],axis='columns')## droping out coloum and one dummy varible 
final


# In[72]:


lin_reg=linear_model.LinearRegression()
lin_reg.fit(final[['area','monroe township','robinsville']],final.price)


# In[73]:


lin_reg.predict([[2800,0,1]])


# In[74]:


lin_reg.predict([[3400,0,0]])


# In[75]:


### ANOTHER WAY TO MAKE LINEAR MODEL 


# In[76]:


from sklearn.linear_model import LinearRegression
model=LinearRegression


# In[77]:


X=final.drop('price',axis='columns') ## training dataset
X


# In[102]:


y= final.price ## dependent varible 
y


# In[103]:


model.fit(X,y) ## dont know whhat issue aries 


# In[80]:


df ## Another method to create dummies varible using sklearn preprocessing onehotcoder method


# In[92]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[98]:


dfle=df
le.fit_transform(dfle.town)


# In[101]:


dfle.town=le.fit_transform(dfle.town)
dfle


# In[104]:


X=df[['town','area']].values## x to be tow dimensional array 
X


# In[105]:


y=dfle.price
y


# In[118]:


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features=[0])


# In[119]:


X=ohe.fit_transform(X).toarray()
X


# In[ ]:





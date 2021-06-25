#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


df=pd.read_csv('export.csv')
df = df.rename(columns = {"GDP Per Capita":"gdp_percapita"})## to remane the coloum name 
df


# In[15]:


x=df['label']
y=df['gdp_percapita']
plt.scatter(x,y)


# In[23]:


plt.xlabel('label')
plt.ylabel('gdp+percapita')
plt.title('canada gdp per capita income')

plt.scatter(df.label,df.gdp_percapita,color='r',marker='+')


# In[26]:


lin_reg=linear_model.LinearRegression()
lin_reg.fit(df[['label']],df.gdp_percapita)


# In[29]:


lin_reg.predict([[2021]])


# In[30]:


lin_reg.predict([[2020]])


# In[36]:


data={'year':[2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032]}
df1=pd.DataFrame(data)
df1


# In[ ]:





# In[ ]:





# In[43]:


p=lin_reg.predict(df1)## predicting ek sasth sare year
p


# In[44]:


df1['newyeargdp']=p ## joing coloum to prediticve year


# In[46]:


df1


# In[48]:


df1.to_csv('newpredictivegdpvalue.csv',index=False)


# In[49]:





# In[51]:


plt.xlabel('label')
plt.ylabel('gdp+percapita')
plt.title('canada gdp per capita income')

plt.scatter(df.label,df.gdp_percapita,color='r',marker='+')
plt.plot(df.label,lin_reg.predict(df[['label']]),color='y')


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('income.csv')
df


# In[9]:


plt.scatter(df.Age,df['Income($)'])
plt.xlabel('age')
plt.ylabel('Income')


# In[20]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted


# 

# In[25]:


df['cluster'] = y_predicted
df


# In[26]:


km.cluster_centers_


# In[32]:


df1=df[df.cluster==0]
df1


# In[60]:


df2=df[df.cluster==1]
df3=df[df.cluster==2]


# In[61]:


plt.scatter(df1.Age,df1['Income($)'],color="g")
plt.scatter(df2.Age,df2['Income($)'],color="b")
plt.scatter(df3.Age,df3['Income($)'],color="r")
plt.xlabel('income($)')
plt.ylabel('age')
plt.legend('123')


# In[62]:


## so scaleing is not done properly that why red cluster is not come perfectly


# In[63]:


## so lets scale it income,age


# In[64]:


scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])
df.head()


# In[65]:


plt.scatter(df.Age,df['Income($)'])


# In[66]:


km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income($)']])
y_predicted


# In[68]:


df['cluster']=y_predicted
df


# In[69]:


km.cluster_centers_


# In[72]:


df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]


# In[75]:


plt.scatter(df1.Age,df1['Income($)'],color="g") ## deke mast a gye cluster )::
plt.scatter(df2.Age,df2['Income($)'],color="b")
plt.scatter(df3.Age,df3['Income($)'],color="r")
plt.xlabel('income($)')
plt.ylabel('age')
plt.legend('123')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroids') ## these are the center 


# In[ ]:



### if we dont the know the value of k than we make a sse model and make a elbow graph 


# In[76]:


sse = []

k_rng = range(1,10) 
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)


# In[77]:


sse


# In[78]:


plt.xlabel('k')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:





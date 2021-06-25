#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


df=load_digits()
df


# In[17]:


dir(df)


# In[18]:


df.data[0] ## image is represnted as one dimesnsional array


# In[19]:


plt.gray() ## use to plot graph 
for i in range(5):
 plt.matshow(df.images[i])


# In[20]:


df.target[0:5]


# In[21]:


from sklearn.model_selection import train_test_split
df1= train_test_split


# In[22]:


X_train,X_test,y_train,y_test= train_test_split(df.data,df.target,train_size=.8)


# In[23]:


len(X_test)


# In[24]:


len(X_train)


# In[29]:


from sklearn.linear_model import LogisticRegression
df1=LogisticRegression()

lin_reg=linear_model.LogisticRegression()
lin_reg.fit(X_train,y_train)


# In[37]:


lin_reg.score(X_test,y_test)


# In[36]:


lin_reg.predict(X_test)


# In[38]:


plt.matshow(df.images[67])


# In[44]:


lin_reg.predict([df.data[67]])


# In[45]:


lin_reg.predict([df.data[1000]])


# In[47]:


df.target[67]


# In[49]:


lin_reg.predict(df.data[0:5])


# In[50]:


### how to chek that your model is not working properly 


# In[53]:


y_predicted=lin_reg.predict(X_test)
from sklearn.metrics import confusion_matrix
c_m= confusion_matrix(y_test,y_predicted)
c_m


# In[55]:


import seaborn as sn
plt.figure(figsize =(10,7))
sn.heatmap(c_m,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')


# In[ ]:


## the 1,2 areas are the place where model is not working good


#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


df=load_digits()
df


# In[18]:


dir(df)


# In[21]:


df.target[:4]


# In[23]:


df1=pd.DataFrame(df.data)
df1


# In[26]:


df1['target']=df['target']
df1


# In[28]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train,X_test,y_train,y_test = train_test_split(df.data,df.target,train_size=.8)


# In[32]:


len(X_train)


# In[41]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)


# 

# In[42]:


model.fit(X_train,y_train)


# In[43]:


model.score(X_test,y_test) ## not bad :)


# In[46]:


df.images


# In[47]:


df.target_names


# In[50]:


plt.gray()### just checking the images 
for i in range(5):
 plt.matshow(df.images[i])


# In[57]:


model.predict([df.data[90]])


# In[58]:


y_predicted=model.predict(X_test)
y_predicted


# In[61]:


from sklearn.metrics import confusion_matrix
cn=confusion_matrix(y_test,y_predicted)## use seaborn libarry to get great vusiulation 
cn


# In[66]:


import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cn,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[ ]:





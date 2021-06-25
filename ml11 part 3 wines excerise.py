#!/usr/bin/env python
# coding: utf-8

# In[165]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[166]:


from sklearn.datasets import load_wine
df=load_wine()


# In[167]:


dir(df)


# In[168]:


df.target_names


# In[169]:


df1=pd.DataFrame(df.data,columns=df.feature_names)
df1


# In[170]:


df1['target']=df['target']
df1


# In[171]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.data, df1.target, test_size=0.3, random_state=100)


# In[172]:



from sklearn.naive_bayes import  MultinomialNB
model1 = MultinomialNB()
model1.fit(X_train,y_train)


# In[173]:


model1.score(X_test,y_test)


# In[174]:


from sklearn.naive_bayes import GaussianNB
model=GaussianNB()


# In[175]:


model.fit(X_train,y_train)


# In[176]:


model.score(X_test,y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
from sklearn.datasets import load_iris 
df= load_iris()


# In[47]:


dir(df)


# In[48]:


df


# In[49]:


df.target


# In[50]:


df.target_names


# In[51]:


df.feature_names


# In[52]:


df1=pd.DataFrame(df.data,columns=df.feature_names)
df1


# In[53]:


df1=pd.DataFrame(df.data,columns=df.feature_names)
df1['target']=df.target
df1


# In[54]:


len(df1)


# In[55]:


df1['target']=df.target
df1


# In[56]:


df1[df1.target==2].head() ## printing table 


# In[ ]:





# In[58]:


df1['flower_name']=df1.target.apply(lambda x: df.target_names[x])
df1


# In[62]:


df0=df1[:50]
df2=df1[50:100]
df3=df1[100:]


# In[63]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[73]:


plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='g',marker="+")
plt.scatter(df2['sepal length (cm)'],df2['sepal width (cm)'])


# In[74]:


plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='g',marker="+")
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'])


# In[79]:


from sklearn.model_selection import train_test_split


# In[83]:


X=df1.drop(['target','flower_name'],axis='columns')
y=df1.target
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=.8)


# In[86]:


len(X_train)


# In[88]:


from sklearn.svm import SVC
model_1=SVC()


# In[89]:


model_1.fit(X_train,y_train)


# In[92]:


model_1.score(X_test,y_test)


# In[93]:


model_1.predict([[6.2,3.4,5.4,2.3]]) ## nice prediction 


# In[99]:


model_2=SVC(C=10) ## regulazition tunning the model
model_2.fit(X_train,y_train)
model_2.score(X_test,y_test)


# In[108]:


model_2=SVC(gamma=.4) ## regulazition tunning the model
model_2.fit(X_train,y_train)
model_2.score(X_test,y_test)


# In[117]:


model_linear_kernal=SVC(kernel='linear')
model_linear_kernal.fit(X_train,y_train)


# In[118]:


model_linear_kernal.score(X_test,y_test)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('titanic.csv')
df


# In[3]:


df1=df.drop(['Name','PassengerId','SibSp','Parch','Embarked','Cabin','Ticket'],axis='columns')
df1


# In[4]:


target=df1.Survived
target


# In[5]:


final=df1.drop(['Survived'] , axis='columns')
final


# In[6]:


final_dummies=pd.get_dummies(final.Sex)
final_dummies


# In[7]:


merged=pd.concat([final,final_dummies],axis='columns')
merged


# In[8]:


titanic=merged.drop(['Sex'],axis='columns')
titanic


# In[9]:


titanic.columns[titanic.isna().any()] ## checking NAN value in any columns


# In[10]:


titanic.Age.head(10)


# In[11]:


titanic.Age=titanic.Age.fillna(titanic.Age.mean()) ## filling all NAN value with mean of that row
titanic.head(10)


# In[12]:


X=titanic[['Pclass','Age','Fare','female','male']]
y=target

from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.8)


# In[14]:


from sklearn.linear_model import LogisticRegression ##logistic regression moddel prediction 
lr=LogisticRegression()


# In[15]:


lin_reg=LogisticRegression()


# In[16]:


lin_reg.fit(X_train,y_train)


# In[17]:


lin_reg.score(X_test,y_test)


# In[18]:


lin_reg.predict([[3,40,35000,0,1]]) ## just an example 


# In[19]:


from sklearn.naive_bayes import GaussianNB ### naive bayes model 
model=GaussianNB()


# In[20]:


model.fit(X_train,y_train)


# In[21]:


model.score(X_test,y_test)


# In[22]:


y_test.head(10)


# In[25]:


model.predict(X_test[:10])## tere is a mistake at 100 and 645 place due to 80% score 


# In[26]:


model.predict_proba(X_test[0:10])


# In[ ]:





# In[ ]:





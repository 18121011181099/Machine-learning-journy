#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df=pd.read_csv('spam.csv')
df.head(10)


# In[11]:


df.shape


# In[13]:


df.groupby('Category').describe()


# In[15]:


df['spam']=df['Category'].apply(lambda x:1 if x=='spam' else 0)
df.head()


# In[57]:


X=df['Message']
y=df['spam']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25)


# In[58]:


from sklearn.feature_extraction.text import CountVectorizer ## this a code for making message 
cv=CountVectorizer()
x = cv.fit_transform(X_train.values,y_train)
x.toarray()[:3]


# In[59]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB
model.fit(x,y_train)## why error


# In[43]:


emails=[
    'hey shivam can we meet together to watch footwall game tomorrow',
    'upto 20% discount on parking, exclusive offer just for you . dont miss this reward'] ## 
emails_count= cv.transform(emails)
model.predict(emails_count)## konsa x argument bhai (:


# In[44]:


x=cv.transform(X_test)
model.score(x,y_test)


# In[51]:


from sklearn.pipeline import Pipeline ## best method to predict spam anaylisis
pp=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
    ])


# In[52]:


pp.fit(X_train,y_train)


# In[54]:


pp.score(X_test,y_test)


# In[56]:


pp.predict(emails)


# In[ ]:





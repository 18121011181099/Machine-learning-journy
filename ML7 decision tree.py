#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('salaries.csv')
df


# In[52]:


df1=df.drop(['salary_more_then_100k'],axis='columns')
target=df['salary_more_then_100k']
df1


# In[29]:


from sklearn.preprocessing import LabelEncoder
lb_company=LabelEncoder()
lb_job=LabelEncoder()
lb_degree=LabelEncoder()


# In[30]:


df1['company_n']=lb_company.fit_transform(df1['company'])
df1['job_n']=lb_job.fit_transform(df1['job'])
df1['degree_n']=lb_degree.fit_transform(df1['degree'])
df1


# In[36]:


df2=df1.drop(['company','job','degree','v',2,0,1],axis='columns')
df2


# In[43]:


from sklearn.preprocessing import LabelEncoder ###  SO THIS IS ALSO APPLICABLE FOR CHANIGING THE WORD VALUES TO NUMBERS VALUES 
lb=LabelEncoder()
df1['x']=lb.fit_transform(df1.company)
df1['y']=lb.fit_transform(df1.job)
df1['z']=lb.fit_transform(df1.degree)
df1



# In[45]:


df2


# In[53]:


from sklearn import tree
md=tree.DecisionTreeClassifier()


# In[55]:


md.fit(df2,target)


# In[56]:


md.score(df2,target) ## because no train_test_split


# In[58]:


md.predict([[2,2,1]])


# 

# In[61]:


md.predict([[2,1,2]])


# In[ ]:





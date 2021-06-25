#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


df=load_digits()
df


# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


X_train,X_test,y_train,y_test = train_test_split(df.data,df.target,train_size=.7)


# In[88]:


len(X_train)


# In[89]:


from sklearn.linear_model import LogisticRegression##1
lg=LogisticRegression()
lg.fit(X_train,y_train)
lg.score(X_test,y_test)


# In[90]:


from sklearn.svm import SVC##2
sv=SVC()
sv.fit(X_train,y_train)
sv.score(X_test,y_test)


# In[91]:


from sklearn.ensemble import RandomForestClassifier##3
rfc=RandomForestClassifier(n_estimators=40)
rfc.fit(X_train,y_train)
rfc.score(X_test,y_test)


# In[92]:


from sklearn.model_selection import  KFold
kf=KFold(n_splits=3)
kf


# In[93]:


for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
 print (train_index,test_index)


# In[94]:


def get_score(model,X_train,X_test,y_train,y_test): ### SIMPLE FORN OF ABOVE 3 MODEL
 model.fit(X_train,y_train)
 return model.score(X_test,y_test)


# In[95]:


get_score(SVC(),X_train,X_test,y_train,y_test ) ## svc model


# In[96]:


get_score(LogisticRegression(),X_train,X_test,y_train,y_test ) ##  logistic regression model


# In[97]:


### creating a K fold model 


# In[98]:


from sklearn.model_selection  import StratifiedKFold
kf=StratifiedKFold(n_splits=3)
score_logistic = []
score_svc =[]
score_RANDOM =[]
for train,test in kf.split(df.data,df.target):
    X_train,X_test,y_train,y_test = df.data[train],df.data[test],df.target[train],df.target[test]
    
    score_svc.append(get_score(SVC(),X_train,X_test,y_train,y_test))
    score_logistic.append(get_score(LogisticRegression(),X_train,X_test,y_train,y_test))
    
    score_RANDOM.append(get_score(RandomForestClassifier(n_estimators=40),X_train,X_test,y_train,y_test))


# In[99]:


score_logistic


# In[100]:


score_svc ### BEST MODEL TO CHOOSE BECAUSWE ITS AVERGAE IS MORE 


# In[101]:


score_RANDOM


# In[102]:


#### we  dont have to do this much big code to solve we have a function cross_val_score(model,data,target) funtion which directly give the result


# In[103]:


from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(),df.data,df.target,cv=3)


# In[104]:


cross_val_score(RandomForestClassifier(),df.data,df.target,cv=3)


# In[105]:


cross_val_score(SVC(),df.data,df.target,cv=3) ## cv=n is used to no of split of data 


# In[106]:


### taking random forest classifier and checking n_estimators value at differnt palce 


# In[107]:


s1=cross_val_score(RandomForestClassifier(n_estimators=10),df.data,df.target,cv=10)
np.average(s1)


# In[108]:


s2=cross_val_score(RandomForestClassifier(n_estimators=20),df.data,df.target,cv=10)
np.average(s2)


# In[109]:


s3=cross_val_score(RandomForestClassifier(n_estimators=30),df.data,df.target,cv=10)
np.average(s3)


# In[110]:


s4=cross_val_score(RandomForestClassifier(n_estimators=50),df.data,df.target,cv=10)
np.average(s4)


# In[ ]:





# In[ ]:





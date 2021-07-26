#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[3]:


#visual representation of data in csv
plt.scatter(df.age,df.bought_insurance,marker='*',color='red')


# In[4]:


# splitting train and test 
from sklearn.model_selection import train_test_split


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.8)


# In[6]:


# values we ar used to test our model
X_test


# In[7]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[8]:


#training for model
model.fit(X_train, y_train)


# In[ ]:







# In[9]:


X_test


# In[10]:


y_predicted = model.predict(X_test)


# In[11]:




model.predict_proba(X_test)


# In[12]:


#accuracy of the model
model.score(X_test,y_test)


# In[13]:


y_predicted


# In[14]:


X_test


# In[15]:




model.coef_


# In[16]:


model.intercept_


# In[17]:


import math
def sigmoid(x):
  return 1 / (1 + math.exp(-x))


# In[18]:


def prediction_function(age):
    z = 0.042 * age - 1.53 
    y = sigmoid(z)
    if(y<0.5):
        y=0
    else:
        y=1
    return y


# In[19]:


age = 35
prediction_function(age)


# In[20]:


age = 43
prediction_function(age)


# In[ ]:





# In[ ]:





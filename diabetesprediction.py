#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[17]:


data = pd.read_csv("C:/Users/Mital/OneDrive/Documents/projects/Dataset MeriSKILL/Project 2 MeriSKILL/diabetes.csv")


# In[18]:


data


# In[19]:


sns.heatmap(data.isnull())


# In[20]:


correlation = data.corr()


# In[21]:


print(correlation)


# In[22]:


import matplotlib.pyplot as plt


# In[24]:


plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[31]:


X= data.drop("Outcome",axis=1)
Y=data['Outcome']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)


# In[34]:


model=LogisticRegression()
model.fit(X_train,Y_train)


# In[35]:


prediction = model.predict(X_test)


# In[36]:


print(prediction)


# In[37]:


accuracy=accuracy_score(prediction,Y_test)


# In[38]:


print(accuracy)


# In[ ]:





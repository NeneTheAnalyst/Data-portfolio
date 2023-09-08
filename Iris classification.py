#!/usr/bin/env python
# coding: utf-8

# In[1]:


# libaraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Load iris dataset
df = pd.read_csv("Downloads/Iris.csv")


# In[3]:


df.head()


# In[4]:


# drop the column Id
df.drop(['Id'], axis = 1, inplace=True)


# In[5]:


# check for missing value(s)
df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


# get the size of the dataset
df.shape


# In[9]:


df.dtypes


# In[10]:


# get the dependent variables and independent variable
X = df.iloc[:, :-1].values # matrix of features
y = df.iloc[:, -1].values # dependent vector


# In[11]:


# Print your feature matrix (X) and dependent variable vector (y)
print("The feature matrix is: ", X)
print("The dependent variable is: ", y)


# In[12]:


# Encodeing the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[13]:


print(y)


# In[14]:


# split the dataset into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# In[15]:


print(X_train)


# In[16]:


print(X_test)


# In[17]:


print(y_train)


# In[18]:


print(y_test)


# In[19]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[20]:


print(X_train)


# In[21]:


print(X_test)


# ## Training the Logistic Regression model on the Training set

# In[22]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 0)
LR.fit(X_train, y_train)


# ## Predicting a new result

# In[23]:


print(LR.predict(sc.transform([[5.4, 3.9, 1.3, 0.4]])))


# ## Predicting the Test set results

# In[24]:


y_pred = LR.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[25]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print(CM)


# ## Computing the accuracy with K-Fold Cross Validation
# 

# In[26]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LR, X = X_train, y = y_train, cv = 10) # 10 is the default number
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# # Conclusion

# The model successfully identified the species of Iris flowers with an accuracy rate of approximately 95%. In simpler terms, when it attempted to determine the species of an Iris flower (such as setosa, versicolor, or virginica), it made the correct prediction in 95 out of 100 instances. This level of accuracy signifies that the model's performance is commendable.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


#librairies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#import dataset
df = pd.read_csv("Downloads/breast_cancer.csv")


# In[4]:


df.head()


# # NOTE:
# The Class column is the target variable or label. It typically indicates whether a sample is benign (non-cancerous) or malignant (cancerous). It have values of 2 and 4, where 2 represent benign and 4 represent malignant.

# In[8]:


#dropping the sample code number
df.drop(['Sample code number'], axis = 1, inplace=True)
# checking for missing values
df.isnull().sum()


# In[9]:


#statistical description
df.describe()


# In[10]:


#size of the dataset
df.shape


# In[11]:


# get the dependent variables and independent variable
X = df.iloc[:,1:-1].values # set of features
y = df.iloc[:, -1].values # dependent variable(vector)


# In[12]:


# Print your feature matrix (X) and dependent variable vector (y)
print("The feature matrix is: ", X)
print("The dependent variable is: ", y)


# In[13]:


# spliting the dataset into training test and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[14]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


print(X_train)


# In[16]:


print(X_test)


# In[17]:


# Training the logistic Regression model on the Trainig set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)


# ### Predicting a new result

# In[20]:


print(lr.predict(sc.transform([[7, 3, 1, 4, 8, 10, 4, 2]])))


# ### Predicting the Test set results

# In[21]:


y_pred = lr.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix
# 

# In[22]:


from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, y_pred)
print(CM)


# ## Computing the accuracy with K-Fold Cross Validation
# 

# In[24]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10) # 10 is the default number
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


# # Conclusion

# The accuracy score indicates that the model can accurately classify whether a breast cancer case is malignant or benign approximately 96.70% of the time. To put it differently, it correctly predicts the outcome for nearly 97 out of every 100 cases it evaluates. This high accuracy underscores the model's effectiveness in distinguishing between these two categories of breast cancer.
# Additionally, the standard deviation of 2.43% indicates that, on average, the model's accuracy may fluctuate by approximately 2.43% from the reported 96.70% accuracy, reflecting some variability in its performance.

# In[ ]:





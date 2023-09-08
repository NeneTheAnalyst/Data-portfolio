#!/usr/bin/env python
# coding: utf-8

# # Medical insurance cost prediction using linear regression
# 
# 

# In[1]:


# LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Data Sets
# 

# In[2]:


df = pd.read_csv("Downloads/insurance.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.describe()


# In[6]:


df.describe(include = 'all')


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.dtypes


# In[10]:


df.isnull().sum()


# ### Exploratory Data Analysis
# 

# In[11]:


# Countplot to get the count of all the gender
plt.figure(figsize=(4,4))
sns.countplot(x='sex', data=df)
plt.title('Gender Distribution')


# In[12]:


plt.figure(figsize=(4,4))
sns.countplot(x='smoker', data = df)
plt.title('Smokers')


# In[13]:


plt.figure(figsize=(4,4))
sns.countplot(x='region', data = df)
plt.title('Region')


# In[14]:


plt.figure(figsize=(4,4))
sns.barplot(x="region", y="charges", data=df)
plt.title('Cost vs Region')


# The plot shows that people from Southeast have a higher medical insurance cost

# In[15]:


plt.figure(figsize=(4,4))
sns.barplot(x="smoker", y="charges", data=df)
plt.title('Cost for Smokers')


# The plot shows that Smokers have a higher medical insurance cost

# In[16]:


plt.figure(figsize=(4,4))
sns.barplot(x="sex", y="charges",hue='smoker', data=df)
plt.title('Cost for Smokers')


# In[17]:


fig, axes = plt.subplots(1,3, figsize =(13, 4), sharey = True)
fig.suptitle('Visualizing Catergorial Columns')
sns.boxenplot(x='smoker', y ='charges', data =df, ax=axes[0])
sns.boxenplot(x='sex', y ='charges', data =df, ax=axes[1])
sns.boxenplot(x='region', y ='charges', data =df, ax=axes[2])


# In[18]:


df[['age', 'bmi', 'children', 'charges']].hist(bins=20, figsize=(8,8), color = 'red')
plt.show()


# ## Coverting categorical variable into numerical format

# In[19]:


df['sex'] = df['sex'].apply({'male':0, 'female':1}.get)
df['smoker'] = df['smoker'].apply({'yes':1, 'no':0}.get)
df['region'] = df['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)


# In[20]:


df.head()


# In[21]:


# using heatmap to visualize the correleation between the variables
plt.figure(figsize = (8, 5))
sns.heatmap(df.corr(), annot = True)
plt.show()


# # Model Development

# In[23]:


X = df.drop(['charges', 'sex'], axis=1)
y = df.charges


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)


# In[25]:


#create linear regression object
lm = LinearRegression()


# In[26]:


# Fit the linear model
lm.fit(X_train, y_train)
Y_hat = lm.predict(X_test)


# In[27]:


#find R^2
from sklearn.metrics import r2_score
print('The R-square is: ', r2_score(y_test, Y_hat))


# In[28]:


# PLOT OF PREDICTED VALUES VS ACTUAL VALUES
plt.scatter(y_test, Y_hat)
plt.xlabel('Y Test')
plt.ylabel('Y Pred')
plt.show()


# # TESTING THE MODEL

# In[29]:


#predicting insurance cost for a new customer
Input = {'age':25, 'bmi':20, 'children':1, 'smoker':0, 'region':2}
index = [0]
df2 = pd.DataFrame(Input, index)
df2


# In[30]:


costPred = lm.predict(df2)
print('The insurance cost of the new customer is: ', costPred)


# In[31]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, Y_hat)
print('The mean square error is: ', mse)


# # Conclusion

# We can draw the inference that individuals from the Southeastern region exhibit elevated medical insurance expenses, and similarly, smokers also experience increased medical insurance costs. Furthermore, the model demonstrates proficiency in forecasting outcomes for new data.

# 

# In[ ]:





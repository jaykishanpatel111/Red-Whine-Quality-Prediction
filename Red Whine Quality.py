#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


cd F:\Dataset\Done projects\Red Whine Quality Prediction


# In[3]:


pwd


# # Read Dataset

# In[4]:


df=pd.read_csv('red-wine-quality.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# # Checking null values in the dataset

# In[8]:


df.isna().sum()


# # Checking Duplicate in data

# In[9]:


df.duplicated().sum()


# In[10]:


df.drop_duplicates(inplace = True)


# In[11]:


df.shape


# # Data Visualization

# In[12]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = df)


# In[13]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df)


# In[14]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'pH', data = df)


# In[15]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)


# In[16]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = df)


# In[17]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = df)


# In[18]:


df.quality.unique()


# In[19]:


df.quality = [1 if each >= 6 else 0 for each in df.quality]
df.quality


# In[20]:


df.quality.value_counts()


# In[22]:


sns.countplot(df.quality)


# # Correlation between each features

# In[23]:


corr_data = df.corr()


# In[24]:


corr_data


# # Ploting heat map of the correlated data

# In[25]:


plt.figure(figsize = (10,5))
sns.heatmap(corr_data, annot = True, cmap = 'RdYlGn')


# # Model Building

# In[26]:


# Create features and target data


# In[27]:


X = df.drop(['quality'], axis= 1)
Y = df.quality.values
col = X.columns


# In[28]:


X.head()


# # Scaling  X(independent) data

# In[29]:


from sklearn.preprocessing import RobustScaler
scaling = RobustScaler()
X = scaling.fit_transform(X)


# In[30]:


X = pd.DataFrame(X,columns=col)
X.head()


# In[32]:


Y


# In[33]:


X.shape, Y.shape


# # Spliting training and testing dataset

# In[34]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.20,random_state=85)


# In[35]:


X_train.shape, Y_train.shape  , X_test.shape, Y_test.shape


# # Creating Random Forest model

# In[36]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 200, random_state = 45)


# In[37]:


# train the model
model.fit( X_train , Y_train.ravel())


# In[38]:


Y_test


# In[39]:


# Predicting values from the model
Y_pred = model.predict(X_test)
Y_pred = np.array([0 if i < 0.5 else 1 for i in Y_pred])
Y_pred


# # Checking accuracy score of our model

# In[40]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score


# In[41]:


def run_model(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train,Y_train.ravel())
    accuracy = accuracy_score(Y_test, Y_pred)
    print("pricison_score: ",precision_score(Y_test, Y_pred))
    print("recall_score: ",recall_score(Y_test, Y_pred))
    print("Accuracy = {}".format(accuracy))
    print(confusion_matrix(Y_test,Y_pred))


# In[42]:


run_model(model, X_train, Y_train, X_test, Y_test)


# In[43]:


cm = confusion_matrix(Y_test, Y_pred)


# In[44]:


# Heatmap of Confusion matrix
sns.heatmap(pd.DataFrame(cm), annot=True)


# # Classification Report

# In[45]:


from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[5]:


#loading the data set
import pandas as pd
data = pd.read_csv('iris.csv')
data.head(10)


# In[31]:


#Statistical Summary


# In[7]:


data.describe()


# In[ ]:


#Data information


# In[9]:


data.info()


# In[ ]:


#Missing values


# In[19]:


print("\nMissing values per column:")
data.isnull().sum()


# In[ ]:


#shape


# In[13]:


data.shape


# In[ ]:


#Detecting Outliers


# In[15]:


# Loop through all numeric columns and check for outliers using IQR
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower) | (data[column] > upper)]
    print(f"{column}: {len(outliers)} outliers")


# In[ ]:


#Replacing Outliers with median


# In[17]:


import pandas as pd

# Loop through all numeric columns
for column in data.select_dtypes(include=['int64', 'float64']).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    # Median of the column
    median = data[column].median()
    
    # Replace outliers with median
    data[column] = data[column].apply(lambda x: median if x < lower or x > upper else x)

print("Outliers replaced with median successfully.")


# In[ ]:


#Data Distribution


# In[23]:


# Distribution of target variable
print("\nClass distribution:")
print(data['species'].value_counts())

# Plot class distribution
sns.countplot(x='species', data=data)
plt.title("Class Distribution")
plt.show()


# In[ ]:


# Histogram and box plots


# In[25]:


# Histograms for each numeric column
data.hist(figsize=(10, 8), bins=20)
plt.suptitle("Histograms of Numeric Features")
plt.show()

# Boxplots
plt.figure(figsize=(12, 8))
for i, column in enumerate(data.select_dtypes(include=['float64', 'int64']).columns, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=data[column])
    plt.title(f"Boxplot of {column}")
plt.tight_layout()
plt.show()


# In[ ]:


#Correlation matrix


# In[27]:


# Correlation matrix
corr = data.corr(numeric_only=True)

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:


# Scatter plots


# In[29]:


# Pairplot (scatter plots with hue = species)
sns.pairplot(data, hue="species", diag_kind="kde")
plt.show()


# In[ ]:





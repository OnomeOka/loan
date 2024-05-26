#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# In[2]:


df=pd.read_csv('loan.csv')


# In[3]:


df.head()


# In[4]:


# missing values
df.isnull().sum()


# In[5]:


#checking for duplicates
df.duplicated()


# In[6]:


#checking for the descriptive statistics like,mean,count etc.
df.describe()


# In[7]:


# to see the occupations of indivisual on the dataset
df.occupation.unique()


# In[8]:


#loan approved by occupation
g1=df[df['loan_status']=='Approved'].groupby('occupation')['occupation'].value_counts()
g1_df=g1.reset_index(drop=True)
g1_df


# In[9]:


#visualization of approved loan
g1_df.plot(kind='bar',x='occupation',y='count')
plt.ylabel('total loan approved')
plt.xlabel('occupation')
plt.show()


# In[10]:


#loan approved by gender
g2=df[df['loan_status']=='Approved'].groupby('gender')['gender'].value_counts()
g2_df=g2.reset_index(drop=True)
g2_df


# In[11]:


#visualization of loan approved by gender
g2 = df[df['loan_status'] == 'Approved'].groupby('gender').size()
g2_df = g2.reset_index(name='count')

plt.pie(g2_df['count'], labels=g2_df['gender'], autopct='%0.1f%%')
plt.show()


# In[12]:


df.head()


# In[13]:


#checking thier education level
df['education_level'].unique()


# In[14]:


# to see rhe educational level that was approved the most
g3=df[df['loan_status']=='Approved'].groupby('education_level')['education_level'].value_counts()
g3


# In[15]:


#visualization of the educational level apporved the more 
plt.pie(g3,labels=g3.index,autopct='%0.1f%%')
plt.title('% of Loan Approved per education level')
plt.show()


# In[16]:


#using histogram to see the income of people which loan was apporved
sns.histplot(df[df['loan_status']=='Approved'].income,kde=True)
plt.xlabel('income')
plt.ylabel('number of loan approved')


# In[17]:


#using histogram to see the credit score of people whic loan was approved
sns.histplot(df[df['loan_status']=='Approved'].credit_score,kde=True,color='green',label='approved')
sns.histplot(df[df['loan_status']=='Demied'].credit_score,kde=True,color='red',label='denied')
plt.xlabel('credit score')
plt.ylabel('count')
plt.show()


# In[18]:


#create a function where the approved loan will be 1 and if not approved 0
def change_val(x):
    if x=='Approved':
        return 1
    else:
        return 0
df['loan_status'] =df['loan_status'].apply(change_val)
df.head()


# In[19]:


def change_val(x):
    if x=='Approved':
        return 1
    else:
        return 0
    
df['loan_status']=df['loan_status'].apply(change_val) 
df.head()


# In[25]:


from sklearn.preprocessing import LabelEncoder
occupation_=LabelEncoder()
education_level_=LabelEncoder()
marital_status_=LabelEncoder()
gender_=LabelEncoder()


# In[27]:


#converting occupation,education_level,marital_status,gender to numerical 
df['occupation'] = occupation_.fit_transform(df['occupation'])
df['education_level'] = occupation_.fit_transform(df['education_level'])
df['marital_status'] = occupation_.fit_transform(df['marital_status'])
df['gender'] = gender_.fit_transform(df['gender'])
df.head()


# In[28]:


# to predict loan status
x=df.drop(['loan_status'],axis=1)
y=df['loan_status']


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)
x_train.head()


# In[30]:


x_train.shape


# In[31]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()



# In[32]:


model.fit(x_train,y_train)


# In[33]:


predicted_val=model.predict(x_test)


# In[34]:


predicted_val


# In[35]:


accuracy=100*model.score(x_test,y_test)
accuracy


# In[37]:


from sklearn.metrics import confusion_matrix


# In[39]:


array=confusion_matrix(y_test,predicted_val)


# In[41]:


sns.heatmap(array, annot=True)
plt.title('confusion matrix')
plt.ylabel('actual value')
plt.xlabel('predicted value')
plt.show()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle


# In[2]:


df_org = pd.read_csv('train.csv')
reqd_cols = ['UID','state','city', 'place','lat','lng','ALand','AWater','pop','male_pop','female_pop','rent_mean','rent_median','rent_stdev','family_mean','family_median','family_stdev','hc_mortgage_mean','hc_mortgage_median',
           'hc_mortgage_stdev','home_equity_second_mortgage','second_mortgage','home_equity', 'debt', 'male_age_mean', 'male_age_median', 'male_age_stdev','female_age_mean', 'female_age_median', 'female_age_stdev',
           'married','separated','divorced']
df = df_org[reqd_cols]


# In[3]:


# Handling Missing Values
# Since the data in the attributes of this dataset are skewed, the missing values are imputed using median
for i in df.columns:
    if df[i].isna().sum()!=0:
        df[i].fillna(df[i].median(),inplace=True)


# In[5]:


df['bad_debt'] = (df['second_mortgage']+df['home_equity']-df['home_equity_second_mortgage'])
df['pop_density'] = df['pop']/df['ALand']
df['median_age'] = (df['male_age_median']+df['female_age_median'])/2
df['pop_bins'] = pd.cut(df['pop'], bins=5, labels=['ver_low','low','medium','high','very_high'])


# In[6]:


def cat_debt(x):
    if x['bad_debt']<0.05:
        return 'Low'
    elif 0.05<x['bad_debt']<0.1:
        return 'Medium'
    else:
        return 'High'
df['bad_debt_cat'] = df.apply(cat_debt, axis=1)


# In[7]:


df['hs_degree'] = df_org['hs_degree']
df['pop_age_median'] = (df['male_age_median']+df['female_age_median'])/2


# In[8]:


fact_list=list()
for i in df_org.columns:
    if df_org[i].dtypes == 'float64':
        fact_list.append(i)
fact_df = df_org[fact_list]
fact_df.head()


# In[9]:


fact_df.drop(['BLOCKID'], axis=1,inplace=True)


# In[10]:


for i in fact_df.columns:
    if fact_df[i].isna().sum()!=0:
        fact_df[i].fillna(fact_df[i].median(),inplace=True)


# In[11]:


fact_df[['County_Id','State']] = df_org[['COUNTYID','state']]


# In[12]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
fact_df['State'] = le.fit_transform(fact_df['State'])
fact_df.head()


# In[13]:


# Data Modelling
# Checking the correlation for Spearmann Correlation
from scipy import stats
correlation_1 = pd.DataFrame(columns=['r','p'])
for col in fact_df.columns.to_list():
        r , p = stats.spearmanr(fact_df['hc_mortgage_mean'], fact_df[col])
        correlation_1.loc[col] = [round(r,3), round(p,30)]

correlation_1.style.set_caption('Mean_Mortgage_Expenditure Correlation Table')


# In[14]:


# Selecting correlation value above
Predictors = list()
for i in range(len(correlation_1.index)):
    if correlation_1['r'][i]>0.4:
        Predictors.append(correlation_1.index[i])
Predictors.remove('hc_mortgage_mean')


# In[15]:


# Training Data-Set
x = fact_df[Predictors]
y = fact_df['hc_mortgage_mean']


# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


# In[17]:


print(x_train.shape)
print(x_test.shape)


# In[18]:


from sklearn.linear_model import LinearRegression
model_new = LinearRegression()
model_new.fit(x_train,y_train)


# In[19]:


pickle.dump(model_new, open('model_new.pkl','wb')) # Saving model to disk


# In[20]:


model_new = pickle.load(open('model_new.pkl','rb'))


# In[21]:


y_test_pred = model_new.predict(x_test)
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test,y_test_pred)
print(f'MAE: {(MAE)}')
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_test_pred)
print(f'RMSE: {np.sqrt(MSE)}')

from sklearn.metrics import mean_absolute_percentage_error
MAPE = mean_absolute_percentage_error(y_test,y_test_pred)
print(f'MAPE: {MAPE}')

from sklearn.metrics import r2_score
R_squared = r2_score(y_test,y_test_pred)
print(f'R_squared: {R_squared}')


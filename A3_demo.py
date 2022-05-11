#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import seaborn as sns

#For date-time
import math
from datetime import datetime
from datetime import timedelta

# Another imports if needs
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose as season
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression 
from sklearn import preprocessing

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten


# In[2]:


def forecasting(X, y):
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(Y), test_size = 0.3, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train,y_train)

    # Get the mean absolute error on the validation data
    predicted_prices = model.predict(X_test)
    MAE = mean_absolute_error(y_test , predicted_prices)
    print('Random forest validation MAE = ', MAE)
    return model, predicted_prices, X_train, X_test, y_train, y_test


# In[3]:


df_store = pd.read_csv('stores.csv') #store data
df_train = pd.read_csv('train.csv') # train set
df_features = pd.read_csv('features.csv')


# In[4]:


df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
df.head(5)


# In[5]:


df.head(150)


# In[6]:


df.drop(['IsHoliday_y'], axis=1,inplace=True)
df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True)
df.shape


# In[7]:


# Removing rows with incorrect (i.e. negative) sales values
df = df.loc[df['Weekly_Sales'] > 0]
df.shape


# In[8]:


df = df.fillna(0)
df.head(5)


# In[9]:


df_encoded = df.copy()
type_group = {'A':1, 'B': 2, 'C': 3}
df_encoded['Type'] = df_encoded['Type'].replace(type_group)
df_encoded.head(5)


# In[10]:


df_encoded['IsHoliday'] = df_encoded['IsHoliday'].astype(bool).astype(int)
df_encoded.head(5)


# In[11]:


df_encoded.drop(['Date'], axis=1,inplace=True)
df_encoded.head(5)

feature_cols = [c for c in df_encoded.columns.to_list() if c not in ["Weekly_Sales"]]
X = df_encoded[feature_cols]
Y = df_encoded['Weekly_Sales']


# In[12]:


df_encoded.head(5)


# In[14]:


model, predicted_test, X_train, X_test, y_train, y_test = forecasting(X, Y)


# In[15]:


np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.2f}'.format})


# In[17]:


for i in range(5):
    print(X_test[i], predicted_test[i], y_test[i])


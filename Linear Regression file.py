#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.style
plt.style.use("classic")

import seaborn as sns

from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv(r'C:\Users\numaa\Downloads\car_mpg.csv')
df.head()


# In[3]:


df.drop('car_name', axis = 1, inplace  = True)
df.head()


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
df


# In[7]:


df = pd.get_dummies(df, columns=['origin'])


# In[8]:


df


# In[9]:


df.describe().transpose()


# In[10]:


df.info()


# In[11]:


df.dtypes


# In[12]:


temp = pd.DataFrame(df.hp.str.isdigit())


# In[13]:


temp[temp['hp'] == False] 


# In[14]:


df.tail(50)


# In[15]:


df = df.replace('?', np.nan) 


# In[16]:


df[df.isnull().any(axis = 1)] 


# In[17]:


df.hp.median()


# In[18]:


df = df.apply(lambda x: x.fillna(x.median()), axis = 0)


# In[19]:


df.dtypes


# In[20]:


df['hp'] = df['hp'].astype('float64')


# In[21]:


df.dtypes


# In[22]:


df.describe().transpose()


# In[23]:


df_attr = df.iloc[:, 0:10]
df_attr.head()


# In[24]:


sns.pairplot(df_attr, diag_kind = 'kde')


# In[25]:


x = df.drop('mpg', axis = 1)
x = x.drop({'origin_america', 'origin_asia', 'origin_europe'}, axis = 1)
y = df[['mpg']]


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.30, random_state = 1)


# In[28]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[29]:


for idx, col_name in enumerate(x_train.columns):
    print("The coefficient for {} is {}".format(col_name, lr.coef_[0][idx]))


# In[30]:


intercept = lr.intercept_[0]
print("The intercept for our model is {}".format(intercept))


# In[31]:


lr.score(x_train, y_train)


# In[32]:


lr.score(x_test, y_test)


# In[43]:


from sklearn.metrics import r2_score


# In[45]:


y_pred_train = lr.predict(x_train)


# In[46]:


r2_score( y_train['mpg'], y_pred_train)


# In[47]:


y_pred_test = lr.predict(x_test)


# In[48]:


r2_score( y_test['mpg'], y_pred_test)


# In[49]:


# ----------------- stats models library -------------------------------------


# In[50]:


stats_lr = pd.concat([x_train,y_train], axis = 1)
stats_lr.head()


# In[51]:


stats_lr.shape


# In[52]:


import statsmodels.formula.api as smf
lm1 = smf.ols(formula = 'mpg ~ cyl+disp+hp+wt+acc+yr+car_type', data = stats_lr).fit() 
lm1.params


# In[53]:


print(lm1.summary()) 


# In[60]:


mse = np.mean((lr.predict(x_test)- y_test)**2)


# In[61]:


import math
math.sqrt(mse)


# In[62]:


lr.score(x_test, y_test)


# In[63]:


y_pred = lr.predict(x_test)


# In[64]:


plt.scatter(y_test['mpg'], y_pred)


# In[65]:


# ----------------------------- iteration 3 --------------------------------------------


# In[66]:


from scipy.stats import zscore

x_train_scaled = x_train.apply(zscore)

x_test_scaled = x_test.apply(zscore)
y_train_scaled = y_train.apply(zscore)
y_test_scaled = y_test.apply(zscore)


# In[67]:


lr2 = LinearRegression()
lr2.fit(x_train_scaled, y_train_scaled)


# In[68]:


for idx, col in enumerate(x_train_scaled.columns):
    print("The coefficient for {} is {}".format(col, lr2.coef_[0][idx]))


# In[69]:


intercept2 = lr2.intercept_[0]
print('The intercept for our model is {}'.format(intercept2))


# In[70]:


lr2.score(x_test_scaled, y_test_scaled)


# In[71]:


mse2 = np.mean((lr2.predict(x_test_scaled)- y_test_scaled)**2)


# In[72]:


import math
math.sqrt(mse2)


# In[73]:


y_pred2 = lr2.predict(x_test_scaled)


# In[74]:


plt.scatter(y_test_scaled['mpg'],y_pred2)


# In[75]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[76]:


vif = [variance_inflation_factor(x.values, ix) for ix in range(x.shape[1])]


# In[77]:


i = 0
for column in x.columns:
    if i<11:
        print(column, '-->', vif[i])
        i = i + 1


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import statsmodels.api as sm
import os
os.chdir('K:/data/DataSience/data_sicence course/ML/farzad/Regression Problem') # set your directory in your local machine


# In[2]:


os.getcwd()


# In[3]:


data = pd.read_csv('CS_04.csv')
data.shape


# In[4]:


data.columns


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.isna().sum()


# In[8]:


data.drop(['Unnamed: 0'], axis = 1, inplace = True)
data.head()


# In[9]:


data2 = data.dropna(subset = ['Salary'], axis = 0, inplace = False)
data2.shape


# In[10]:


data2.describe()


# ### Histogram Plot for All Features

# In[11]:


data2.columns


# In[12]:


var_ind = list(range(13)) + list(range(15, 19))
len(var_ind)


# In[13]:


var_ind


# In[14]:


data2.iloc[:, var_ind].info()


# In[15]:


plot = plt.figure(figsize = (12, 6))
plot.subplots_adjust(hspace = 0.5, wspace = 0.5)
for i in range(1, 18):
    a = plot.add_subplot(3, 6, i)
    a.hist(data2.iloc[:, var_ind[i - 1]], alpha = 0.7)
    a.title.set_text(data2.columns[var_ind[i - 1]])


# In[16]:


# Box Plot of Salary
plt.boxplot(data2.Salary, showmeans = True)
plt.title('Boxplot of Salary')


# In[17]:


cor_table = round(data2.iloc[:, var_ind].corr(method = 'pearson'), 2)
cor_table


# In[18]:


plt.figure(figsize = (12, 8))
sns.heatmap(cor_table, annot = True)


# #### Scatter Plot Between Target Value and Features

# In[19]:


#Scatter Plot
var_ind = list(range(13)) + list(range(15, 18))
plot = plt.figure(figsize = (12, 12))
plot.subplots_adjust(hspace = 0.8, wspace = 0.5)
for i in range(1, 17):
    a = plot.add_subplot(4, 4, i)
    a.scatter(x = data2.iloc[: , var_ind[i - 1]], y = data2.iloc[: , 18], alpha = 0.5)
    a.title.set_text('Salary vs. ' + data2.columns[var_ind[i - 1]])


# In[20]:


targ_x = 'Errors'
targ_y = 'Salary'
print(f'{data2[targ_x].count()} == {data2[targ_x].size} if these numbers are not equal then we have missing value')
print('------------------------------------')
print(f'The min of {targ_x} is: {data2[targ_x].min()}')
print('------------------------------------')
print(f'The max of {targ_x} is: {data2[targ_x].max()}')
print('------------------------------------')
print(f'The quantile of {targ_x} is:\n{data2[targ_x].quantile(q = [.1, .25, .5, .75, .9, .95])}')
print('------------------------------------')
print(f'The mean of {targ_x} is: {data2[targ_x].mean()}')


plt.figure(figsize=(15, 10), dpi = 200)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)


plt.subplot(221)
sns.boxplot(x = data2[targ_x]);
plt.title(f'boxplot of {targ_x}')

plt.subplot(222)
sns.distplot(data2[targ_x], kde = False);
plt.title(f'Histogram of {targ_x}')


plt.subplot(223)
plt.scatter(x = data2[targ_x], y = data2[targ_y], alpha = 0.5);
plt.title(f'{targ_y} vs {targ_x}')
plt.xlabel(targ_x)
plt.ylabel(targ_y)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)



plt.subplot(224)
plt.scatter(x = data2[targ_x], y = data2[targ_y]);
plt.title(f'{targ_y} vs log({targ_x})')
plt.xlabel(targ_x)
plt.ylabel(targ_y)
plt.subplots_adjust(hspace = 0.5, wspace = 0.5)


# ### Categorical Variables

# In[21]:


data2['League'].value_counts()


# In[22]:


data2['Division'].value_counts()


# In[23]:


data2['NewLeague'].value_counts()


# ### Data Preparation

# In[24]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(data2, test_size = 0.2, random_state = 1234)
print(f'The Shape of train is: {train.shape}')
print(f'The Shape of test is: {test.shape}')


# In[25]:


train.describe()


# In[26]:


test.describe()


# In[27]:


final_dataset_train = pd.get_dummies(train,drop_first=True)
final_dataset_train.head()


# In[28]:


X_train = final_dataset_train.iloc[:, list(range(0,16)) + list(range(17, 20))]
y_train = final_dataset_train.iloc[:, 16]


# In[29]:


X_train = sm.add_constant(X_train)


# In[30]:


y_train


# In[31]:


final_dataset_test = pd.get_dummies(test, drop_first=True)
final_dataset_test.head()


# In[32]:


X_test = final_dataset_test.iloc[:, list(range(0,16)) + list(range(17, 20))]
y_test = final_dataset_test.iloc[:, 16]


# In[33]:


X_test = sm.add_constant(X_test)


# In[34]:


y_test


# # Bulid Models
# 

# ## Model1: Linear Regression

# In[35]:


import statsmodels.api as sm
model1 = sm.OLS(y_train, X_train).fit()
model1.summary()


# In[36]:


#Check Assumptions of Regression
#Normality of residuals

#Plot histogram of residuals
sns.distplot(model1.resid, hist = True, 
             kde = True, color = 'green',
             bins = np.linspace(min(model1.resid), max(model1.resid), 30))


# In[37]:


#QQ-plot
qqplot_model1 = sm.qqplot(model1.resid, line = 's')
plt.show()


# In[38]:


#Residuals vs. Fitted Values
sns.regplot(x = model1.fittedvalues, y = model1.resid, lowess = True, 
                       scatter_kws = {"color": "black"}, line_kws = {"color": "red"})
plt.xlabel('Fitted Values', fontsize = 12)
plt.ylabel('Residuals', fontsize = 12)
plt.title('Residuals vs. Fitted Values', fontsize = 12)
plt.grid()


# In[39]:


#Check Cook's distance
sum(model1.get_influence().summary_frame().cooks_d > 1)


# In[40]:


#Check Multicollinearity
#Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)

calc_vif(X_train.iloc[:, 1:])
#If VIF > 10 then multicollinearity is high


# In[41]:


y_pred_model1 = model1.predict(X_test)
abs_err_model1 = np.abs(y_pred_model1 - y_test)


# In[42]:


#Absolute error mean, median, sd, IQR, max, min
from scipy.stats import iqr
model_comp = pd.DataFrame({'Mean of AbsErrors':    abs_err_model1.mean(),
                           'Median of AbsErrors' : abs_err_model1.median(),
                           'SD of AbsErrors' :     abs_err_model1.std(),
                           'IQR of AbsErrors':     iqr(abs_err_model1),
                           'Min of AbsErrors':     abs_err_model1.min(),
                           'Max of AbsErrors':     abs_err_model1.max()}, index = ['Model1: Linear-Ragression'])
model_comp


# In[43]:


y_pred = y_pred_model1
y_test = y_test
model = 'model1_ClassicalRegression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.title(f'{model}');
plt.show();


# In[44]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model1))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model1)))


# 

# In[45]:


from scipy.stats import boxcox
box_results = boxcox(y_train, alpha = 0.05)
box_results


# In[46]:


logy_train = np.log(y_train)


# In[47]:


import statsmodels.api as sm
model1_box = sm.OLS(logy_train, X_train).fit()
model1_box.summary()


# In[48]:


X_train.drop('const', axis = 1, inplace = True)
X_test.drop('const', axis = 1, inplace = True)


# ### Model2: Ridge Regression

# In[49]:


from sklearn.linear_model import Ridge, RidgeCV


# In[50]:


lambda_grid = 10 ** np.linspace(2, -5, num = 100)
lambda_grid;


# In[51]:


lambda_grid = np.linspace(.2, .36, num = 100)
lambda_grid


# In[52]:


ridge_cv = RidgeCV(alphas = lambda_grid, cv = 10, normalize = True)
ridge_cv.fit(X_train, logy_train)
ridge_cv.alpha_


# In[53]:


model2 = Ridge(normalize = True, alpha = ridge_cv.alpha_)
model2.fit(X_train, logy_train)
y_pred_model2 = model2.predict(X_test)
y_pred_model2 = np.exp(y_pred_model2)
abs_err_model2 = np.abs(y_pred_model2 - y_test)


# In[54]:


sns.distplot(y_test - y_pred_model2)


# In[55]:


plt.scatter(y_test , y_pred_model2)


# In[56]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model2))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model2)))


# In[57]:


from sklearn.metrics import r2_score

y_pred = y_pred_model2
y_test = y_test
model = 'model 2 Ridge Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[58]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model2.mean(),
                                             'Median of AbsErrors' : abs_err_model2.median(),
                                             'SD of AbsErrors' :     abs_err_model2.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model2),
                                             'Min of AbsErrors':     abs_err_model2.min(),
                                             'Max of AbsErrors':     abs_err_model2.max()}, index = ['Model2: Ridge Regression']), 
                               ignore_index = False)

model_comp


# ### Model3: Lasso Regression

# In[59]:


from sklearn.linear_model import Lasso, LassoCV


# In[60]:


lambda_grid = 10 ** np.linspace(2, -5, num = 100)


# In[61]:


lambda_grid =  np.linspace(0.0001, 0.01, num = 100)


# In[62]:


#K-fold Cross Validation to Choose the Best Model
lassocv = LassoCV(alphas = lambda_grid, cv = 10, normalize = True)
lassocv.fit(X_train, logy_train)
lassocv.alpha_


# In[63]:


model3 = Lasso(alpha = lassocv.alpha_, normalize = True)
model3.fit(X_train, logy_train)
y_pred_model3 = model3.predict(X_test)
y_pred_model3 = np.exp(y_pred_model3)
abs_err_model3 = np.abs(y_pred_model3 - y_test)


# In[64]:


sns.distplot(y_test - y_pred_model3)


# In[65]:


plt.scatter(y_test , y_pred_model3)


# In[66]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model3))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model3))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model3)))


# In[67]:


from sklearn.metrics import r2_score

y_pred = y_pred_model3
y_test = y_test
model = 'model 3 Lasso Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[68]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model3.mean(),
                                             'Median of AbsErrors' : abs_err_model3.median(),
                                             'SD of AbsErrors' :     abs_err_model3.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model3),
                                             'Min of AbsErrors':     abs_err_model3.min(),
                                             'Max of AbsErrors':     abs_err_model3.max()}, index = ['Model3: Lasso Regression']), 
                               ignore_index = False)

model_comp


# ### Model4: KNN

# In[69]:


from sklearn.neighbors import KNeighborsRegressor
model4 = KNeighborsRegressor(n_neighbors = 2, metric = 'minkowski', p = 2)
model4.fit(X_train, logy_train)
y_pred_model4 = model4.predict(X_test)
y_pred_model4 = np.exp(y_pred_model4)
abs_err_model4 = np.abs(y_pred_model4 - y_test)


# In[70]:


sns.distplot(y_test - y_pred_model4)


# In[71]:


plt.scatter(y_test , y_pred_model4)


# In[72]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model4))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model4))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model4)))


# In[73]:


from sklearn.metrics import r2_score

y_pred = y_pred_model4
y_test = y_test
model = 'model4 KNN Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[74]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model4.mean(),
                                             'Median of AbsErrors' : abs_err_model4.median(),
                                             'SD of AbsErrors' :     abs_err_model4.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model4),
                                             'Min of AbsErrors':     abs_err_model4.min(),
                                             'Max of AbsErrors':     abs_err_model4.max()}, index = ['Model4: KNN Regression']), 
                               ignore_index = False)

model_comp


# In[75]:


from sklearn.model_selection import cross_val_score
n_neighbors = np.arange(1, 20)
cv = 4
cvscores = np.empty((len(n_neighbors), 2))
counter = -1
for k in n_neighbors:
    counter += 1
    knn = KNeighborsRegressor(n_neighbors = k, metric = 'minkowski', p = 2)
    cvscores[counter, :] = np.array([k, np.mean(cross_val_score(knn, X_train, y_train, cv = cv))])
    
cvscores[np.argmax(cvscores[:, 1]), :]


# ### Model5: Decision Tree

# In[76]:


from sklearn.tree import DecisionTreeRegressor
model5 = DecisionTreeRegressor(max_depth = 15, min_samples_leaf = 13, ccp_alpha = 0.0001)
model5.fit(X_train, logy_train)
y_pred_model5 = model5.predict(X_test)
y_pred_model5 = np.exp(y_pred_model5)
abs_err_model5 = np.abs(y_pred_model5 - y_test)


# In[77]:


from sklearn.model_selection import RandomizedSearchCV

max_depth = [int(x) for x in np.linspace(1, 20, num = 15)]
min_samples_leaf = np.arange(15)
ccp_alpha = [0.001, 0.01, 0.1, 0.0001]
objective = ['reg:squarederror']

random_grid = {'max_depth': max_depth, 
               'min_samples_leaf': min_samples_leaf, 
               'ccp_alpha': ccp_alpha}

decision = RandomizedSearchCV(estimator = model5, param_distributions = random_grid, scoring = 'neg_mean_squared_error', n_iter = 10, cv = 8, random_state = 40, n_jobs = 1)
decision.fit(X_train, logy_train)
decision.best_params_


# In[78]:


sns.distplot(y_test - y_pred_model5)


# In[79]:


plt.scatter(y_test , y_pred_model5)


# In[80]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model5))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model5))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model5)))


# In[81]:


from sklearn.metrics import r2_score

y_pred = y_pred_model5
y_test = y_test
model = 'model5 Decision Tree Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[82]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model5.mean(),
                                             'Median of AbsErrors' : abs_err_model5.median(),
                                             'SD of AbsErrors' :     abs_err_model5.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model5),
                                             'Min of AbsErrors':     abs_err_model5.min(),
                                             'Max of AbsErrors':     abs_err_model5.max()}, index = ['Model5: Decision Tree Regression']), 
                               ignore_index = False)

model_comp


# ### Model6: Bagging

# In[83]:


from sklearn.ensemble import BaggingRegressor
model6 = BaggingRegressor(n_estimators = 500, oob_score = True)
model6.fit(X_train, logy_train)
y_pred_model6 = model6.predict(X_test)
y_pred_model6 = np.exp(y_pred_model6)
abs_err_model6 = np.abs(y_pred_model6 - y_test)


# In[84]:


sns.distplot(y_test - y_pred_model6)


# In[85]:


plt.scatter(y_test , y_pred_model6)


# In[86]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model6))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model6))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model6)))


# In[87]:


from sklearn.metrics import r2_score

y_pred = y_pred_model6
y_test = y_test
model = 'model6 Bagging Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[88]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model6.mean(),
                                             'Median of AbsErrors' : abs_err_model6.median(),
                                             'SD of AbsErrors' :     abs_err_model6.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model6),
                                             'Min of AbsErrors':     abs_err_model6.min(),
                                             'Max of AbsErrors':     abs_err_model6.max()}, index = ['Model6: Bagging Regression']), 
                               ignore_index = False)

model_comp


# ### Model7: Random Forest

# In[89]:


from sklearn.ensemble import RandomForestRegressor
model7 = RandomForestRegressor(n_estimators = 246,
                               oob_score = True, 
                               max_features = 'sqrt', 
                               max_depth = 26, 
                               min_samples_split = 20, 
                               min_samples_leaf = 1)

model7.fit(X_train, logy_train)
y_pred_model7 = model7.predict(X_test)
y_pred_model7 = np.exp(y_pred_model7)
abs_err_model7 = np.abs(y_pred_model7 - y_test)


# In[90]:


from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 40)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 40, num = 6)]
min_samples_split = [2, 5, 10, 15, 20,25,30,35,40,100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf_random = RandomizedSearchCV(estimator = model7, 
                               param_distributions = random_grid, 
                               scoring='neg_mean_squared_error',
                               n_iter = 10, 
                               cv = 5, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = 1)

rf_random.fit(X_train, logy_train);
rf_random.best_params_


# In[91]:


sns.distplot(y_test - y_pred_model7)


# In[92]:


plt.scatter(y_test , y_pred_model7)


# In[93]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model7))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model7))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model7)))


# In[94]:


from sklearn.metrics import r2_score

y_pred = y_pred_model7
y_test = y_test
model = 'model7 Random Forest Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[95]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model7.mean(),
                                             'Median of AbsErrors' : abs_err_model7.median(),
                                             'SD of AbsErrors' :     abs_err_model7.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model7),
                                             'Min of AbsErrors':     abs_err_model7.min(),
                                             'Max of AbsErrors':     abs_err_model7.max()}, index = ['Model7: Random Forest Regression']), 
                               ignore_index = False)

model_comp


# ### Model8: Gradient Boost Regression

# In[96]:


from sklearn.ensemble import GradientBoostingRegressor
model8 = GradientBoostingRegressor(learning_rate = 0.2, 
                                   max_depth = 6, 
                                   min_samples_leaf = 3, 
                                   n_estimators = 1000, 
                                   subsample = 1)

model8.fit(X_train, logy_train)
y_pred_model8 = model8.predict(X_test)
y_pred_model8 = np.exp(y_pred_model8)
abs_err_model8 = np.abs(y_pred_model8 - y_test)

from sklearn.model_selection import RandomizedSearchCV

learning_rate = [0.01,0.1, 0.2, 0.3, 0.4, 0.5]
n_estimators = [500, 600, 700, 800, 900, 1000, 1100, 1200]
subsample = [1, 2, 3, 4]
max_depth = [2, 3, 4, 5, 6, 7]
min_samples_leaf = [3, 4, 5, 6, 7, 8]


random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_leaf': min_samples_leaf, 
               'learning_rate':learning_rate, 
                'subsample': subsample}

rf_random = RandomizedSearchCV(estimator = model8, 
                               param_distributions = random_grid, 
                               scoring='neg_mean_squared_error', 
                               n_iter = 10, 
                               cv = 5, 
                               verbose=2, 
                               random_state=42, 
                               n_jobs = 1)

rf_random.fit(X_train, logy_train);
rf_random.best_params_
# In[97]:


sns.distplot(y_test - y_pred_model8)


# In[98]:


plt.scatter(y_test , y_pred_model8)


# In[99]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model8))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model8))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model8)))


# In[100]:


from sklearn.metrics import r2_score

y_pred = y_pred_model8
y_test = y_test
model = 'model8 Gradient Boost Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[101]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model8.mean(),
                                             'Median of AbsErrors' : abs_err_model8.median(),
                                             'SD of AbsErrors' :     abs_err_model8.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model8),
                                             'Min of AbsErrors':     abs_err_model8.min(),
                                             'Max of AbsErrors':     abs_err_model8.max()}, index = ['Model8: Gradient Boost Regression']), 
                               ignore_index = False)

model_comp


# ### Model9: XGBoost 

# In[102]:


from xgboost import XGBRegressor
from scipy.stats import uniform, randint


# In[103]:


{'gamma': 0.19993048585762774,
 'learning_rate': 0.04399969896408463,
 'max_depth': 5,
 'n_estimators': 115,
 'subsample': 0.6931085361721216}


# In[104]:


model9 = XGBRegressor(gamma = 0.19993048585762774, 
                      learning_rate = 0.04399969896408463, 
                      max_depth = 5, 
                      n_estimators = 115, 
                      subsample = 0.6931085361721216, 
                      colsample_bytree = 0.39, 
                      reg_alpha = 0.07, 
                      reg_lambda = 0.055)
model9.fit(X_train, logy_train)
y_pred_model9 = model9.predict(X_test)
y_pred_model9 = np.exp(y_pred_model9)
abs_err_model9 = np.abs(y_pred_model9 - y_test)

params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

xgb = RandomizedSearchCV(estimator = model9, 
                         param_distributions = params,
                         scoring='neg_mean_squared_error',
                         n_iter = 10, 
                         cv = 5, 
                         verbose=2, 
                         random_state=42, 
                         n_jobs = 1)

xgb.fit(X_train, logy_train);
xgb.best_params_
# In[105]:


sns.distplot(y_test - y_pred_model9)


# In[106]:


plt.scatter(y_test , y_pred_model9)


# In[107]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model9))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model9))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model9)))


# In[108]:


from sklearn.metrics import r2_score

y_pred = y_pred_model9
y_test = y_test
model = 'model9 XGradient Boost Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[109]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model9.mean(),
                                             'Median of AbsErrors' : abs_err_model9.median(),
                                             'SD of AbsErrors' :     abs_err_model9.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model9),
                                             'Min of AbsErrors':     abs_err_model9.min(),
                                             'Max of AbsErrors':     abs_err_model9.max()}, index = ['Model9: XGradient Boost Regression']), 
                               ignore_index = False)

model_comp


# ### Model10: CatBoost 

# In[124]:


from catboost import CatBoostRegressor
model10 = CatBoostRegressor(learning_rate = 0.03, 
                            l2_leaf_reg = 5, 
                            depth = 10, verbose = 0)
model10.fit(X_train, logy_train);
y_pred_model10 = model10.predict(X_test)
y_pred_model10 = np.exp(y_pred_model10)
abs_err_model10 = np.abs(y_pred_model10 - y_test)

grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}

cb = RandomizedSearchCV(estimator = model10, 
                        param_distributions = grid,
                        scoring='neg_mean_squared_error',
                        n_iter = 10, 
                        cv = 5, 
                        verbose=2, 
                        random_state=42, 
                        n_jobs = 1)

cb.fit(X_train,logy_train);
cb.best_params_
# In[111]:


sns.distplot(y_test - y_pred_model10)


# In[112]:


plt.scatter(y_test , y_pred_model10)


# In[113]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model10))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model10))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model10)))


# In[114]:


from sklearn.metrics import r2_score

y_pred = y_pred_model10
y_test = y_test
model = 'model10 CatBoost Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[115]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model10.mean(),
                                             'Median of AbsErrors' : abs_err_model10.median(),
                                             'SD of AbsErrors' :     abs_err_model10.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model10),
                                             'Min of AbsErrors':     abs_err_model10.min(),
                                             'Max of AbsErrors':     abs_err_model10.max()}, index = ['Model10: CatBoost Regression']), 
                               ignore_index = False)

model_comp


# ### Model11: LightBoost

# In[116]:


from lightgbm import LGBMRegressor
model11 = LGBMRegressor(gamma = 0.22007624686980065,
                        learning_rate = 0.06661147045343364,
                        max_depth = 2, 
                        n_estimators = 107, 
                        subsample = 0.6137554084460873)
model11.fit(X_train, logy_train)
y_pred_model11 = model11.predict(X_test)
y_pred_model11 = np.exp(y_pred_model11)
abs_err_model11 = np.abs(y_pred_model11 - y_test)

params = {
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3), # default 0.1 
    "max_depth": randint(2, 6), # default 3
    "n_estimators": randint(100, 150), # default 100
    "subsample": uniform(0.6, 0.4)
}

lb = RandomizedSearchCV(estimator = model11, 
                        param_distributions = params,
                        scoring='neg_mean_squared_error',
                        n_iter = 10, 
                        cv = 5, 
                        verbose=2, 
                        random_state=42, 
                        n_jobs = 1)
lb.fit(X_train, logy_train);
lb.best_params_
# In[117]:


sns.distplot(y_test - y_pred_model11)


# In[118]:


plt.scatter(y_test , y_pred_model11)


# In[119]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred_model11))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_model11))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_model11)))


# In[120]:


from sklearn.metrics import r2_score

y_pred = y_pred_model11
y_test = y_test
model = 'model11 LightBoost Regression'

plt.figure(figsize=(15, 15), dpi = 150);
plt.subplots_adjust(hspace = 0.25, wspace = 0.25);

plt.subplots(figsize=(16,6));
plt.scatter(x = y_test, y = y_pred, color = 'blue');
#Add 45 degree line
xp = np.linspace(y_test.min(), y_test.max(), 100)
plt.plot(xp, xp, 'k', alpha = 0.9, linewidth = 2, color = 'red')
plt.xlabel('Actual Value');
plt.ylabel('Predicted Value');

plt.title(f'{model}');
plt.subplots(figsize=(17,6));

x_points = list(range(len(y_test)));
plt.plot(x_points, y_test, label='y_real', color = 'blue');
plt.plot(x_points, y_pred, label='y_predict', color = 'red');
plt.legend();
plt.grid()
plt.title(f'{model}');
plt.show();


# In[121]:


model_comp = model_comp.append(pd.DataFrame({'Mean of AbsErrors':    abs_err_model11.mean(),
                                             'Median of AbsErrors' : abs_err_model11.median(),
                                             'SD of AbsErrors' :     abs_err_model11.std(),
                                             'IQR of AbsErrors':     iqr(abs_err_model11),
                                             'Min of AbsErrors':     abs_err_model11.min(),
                                             'Max of AbsErrors':     abs_err_model11.max()}, index = ['Model11: LightBoost Regression']), 
                               ignore_index = False)

model_comp


# In[123]:


# Seeing The Resualt you can choose a model :
# I think Model9 and Model6 is Good in this case


# In[ ]:





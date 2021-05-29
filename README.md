# Data Science Project 1: Hitters Project
## objective: Regression Analysis -> predict the salary of hitters

### in this project, we want to predict the salary of hitters in hitters dataset with a different approach of modern regression such as:
- linear regression
- ridge regression
- lasso regression
- KNN regression
- decision tree regression
- bagging regression
- random forest regression
- gradient boost regression(GB)
- eXtrem Gradient Boost(XGB)
- CatBoost
- LightBoost


preview:
</br>
for full code, download the ipynb file and the dataset
</br>

## Python Code:
</br>
import numpy as np
</br>
import pandas as pd
</br>
import matplotlib.pyplot as plt
</br>
import seaborn as sns
</br>
from sklearn import metrics
</br>
import statsmodels.api as sm
</br>
from sklearn.linear_model import Ridge, RidgeCV
</br>
from sklearn.linear_model import Lasso, LassoCV
</br>
from sklearn.neighbors import KNeighborsRegressor
</br>
from sklearn.tree import DecisionTreeRegressor
</br>
from sklearn.ensemble import BaggingRegressor
</br>
from sklearn.ensemble import RandomForestRegressor
</br>
from sklearn.ensemble import GradientBoostingRegressor
</br>
from xgboost import XGBRegressor
</br>
from catboost import CatBoostRegressor
</br>
from lightgbm import LGBMRegressor
</br>
import os
</br>
os.chdir('../Regression Problem') # set your directory in your local machine
</br>

data = pd.read_csv('CS_04.csv')
</br>
data.shape
</br>
...
</br>

if there is any problem, comment for me

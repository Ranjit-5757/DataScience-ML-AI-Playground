import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

data=pd.read_csv(r"D:\Full Stack Data Science\4 Sep (Multiple Regression)\multiple_linear_regression_dataset.csv")
data

X=data.iloc[:,:-1]
X

y=data.iloc[:,2]
y

# Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
y_train_pred=model.predict(X_train)

import statsmodels.formula.api as sm
X=np.append(arr=np.ones((20,1)).astype(int),values=X,axis=1)

import statsmodels.api as sm
X_opt=X[:,[0,1,2]]

#Ordinary Least Square
ols=sm.OLS(endog=y,exog=X_opt).fit()
ols.summary()

len(X_test) == len(y_test)

plt.scatter(X_test,y_test)

plt.plot(X_train,y_train_pred,color='blue')













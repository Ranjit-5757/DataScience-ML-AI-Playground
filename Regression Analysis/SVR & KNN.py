import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"D:\Full Stack Data Science\8 Sep (KNN)\8th\EMP SAL.csv")
data

X=data.iloc[:,1:2].values
y=data.iloc[:,2].values

## Fitting Support Vector Regression
from sklearn.svm import SVR
regressor=SVR()
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
#******************************************
regressor=SVR(kernel='linear',gamma='scale')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr

#*********************************************************************************
regressor=SVR(kernel='poly',gamma='scale')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr

# Degree 4
regressor=SVR(kernel='poly',degree=4,gamma='scale')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr

# Degree 5
regressor=SVR(kernel='poly',degree=5,gamma='scale')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
#**********************************************************************************
regressor=SVR(kernel='sigmoid',gamma='scale')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
#*********************************************************************************

regressor=SVR(kernel='rbf',gamma='auto')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr

#*********************************************************************************
regressor=SVR(kernel='linear',gamma='auto')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
#*********************************************************************************
regressor=SVR(kernel='poly',gamma='auto')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
#*********************************************************************************
regressor=SVR(kernel='sigmoid',gamma='auto')
regressor.fit(X,y)

y_pred_svr=regressor.predict([[6.5]])
y_pred_svr
#*************************************************************************************

# K Nearst Neighbour
from sklearn.neighbors import KNeighborsRegressor
regressor=KNeighborsRegressor()
regressor.fit(X,y)
y_pred_knn=regressor.predict([[6.5]])
y_pred_knn
#****************************************************************
regressor=KNeighborsRegressor(n_neighbors=6,weights='distance',algorithm='brute')
regressor.fit(X,y)
y_pred_knn=regressor.predict([[6.5]])
y_pred_knn
#************************************************************************************
regressor=KNeighborsRegressor(n_neighbors=6,weights='uniform',algorithm='kd_tree')
regressor.fit(X,y)
y_pred_knn=regressor.predict([[6.5]])
y_pred_knn
#*************************************************************************************
regressor=KNeighborsRegressor(n_neighbors=3,weights='uniform',algorithm='ball_tree')
regressor.fit(X,y)
y_pred_knn=regressor.predict([[6.5]])
y_pred_knn
#*****************************************************************************************
regressor=KNeighborsRegressor(n_neighbors=6,weights='distance',algorithm='ball_tree')
regressor.fit(X,y)
y_pred_knn=regressor.predict([[6.5]])
y_pred_knn




























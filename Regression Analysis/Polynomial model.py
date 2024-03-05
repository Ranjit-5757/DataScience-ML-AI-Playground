import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset

salary=pd.read_csv(r"D:\Full Stack Data Science\7 Sep (polynomial Regression)\emp_sal.csv")
salary

# Split data into dependent and independent variable
X=salary.iloc[:,1:2].values
y=salary.iloc[:,2].values

#when  the data is very  less we can build the model or predict without train _test_splt 

# Simple Linear Model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

y_pred=lin_reg.predict([[6.5]])
y_pred

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures()
poly_reg.fit(X,y)

X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
#**********************************************************************
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly, y)
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
#************************************************************************
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
#***************************************************************************
poly_reg=PolynomialFeatures(degree=5)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
#**************************************************************************
poly_reg=PolynomialFeatures(degree=6)
X_poly=poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

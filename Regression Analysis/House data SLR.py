# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Import Dataset
house=pd.read_csv(r"D:\Full Stack Data Science\1 Sep (Simple Linear Regression)\1st\SLR - Practicle\House_data.csv")
house

# Splitting Dependent(y) and Independent(X) Variables
price=house.iloc[:,2].values
y=np.array(price)
y

sqfit=house.iloc[:,5].values
X=np.array(sqfit).reshape(-1,1)
X

##Flitting data into train and test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


#Fitting Simple Linear Model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)


#predicting the price
y_pred=regressor.predict(X_test)


# To check overfitting (low bias and high variance)
bias=regressor.score(X_train, y_train)
bias

# To check underfitting (high bias and low variance)
variance=regressor.score(X_test, y_test)
variance

# Slope is generated from Linear Regression Algorithm
m=regressor.coef_
m

# Intercept also generate by model
c=regressor.intercept_
c


# Future forecast
y=273*770-29315.41
y

# Visualize the training data
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.show()

# Visualize the test result
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.show()


















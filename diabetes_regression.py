# Intro to linear regression practice 
# Date: 11/06/2024  
# Description: This was built coding along side a tutorial video and is not original work. 

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from sklearn import datasets

# Load dataset
diabetes = datasets.load_diabetes()

print(diabetes.DESCR)

# Assign data and target variables 
X = diabetes.data
Y = diabetes.target

# Split the data into training and learning datasets
# 80/20 data split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_train.shape)

# Build linear regression
model = linear_model.LinearRegression()

# Build training model
model.fit(X_train, Y_train)

# Make a prediction
Y_pred = model.predict(X_test)

# Print model performance
print('Coefficients:', model.coef_)
print('Intercepts:', model.intercept_)
print('Mean squared error (MSE): %.2f' % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f' % r2_score(Y_test, Y_pred))

print(diabetes.feature_names)

# show data
plt.scatter(Y_test, Y_pred, alpha=0.5)
plt.title("Diabetes Linear Regression")
plt.show()
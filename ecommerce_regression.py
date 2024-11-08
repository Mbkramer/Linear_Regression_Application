# Intro to linear regression practice 
# Date: 11/06/2024  
# Description: This was built coding along side a tutorial video and is not original work. 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn import datasets

df = pd.read_csv("Datasets/ecommerce_customers.csv")

print(df.head())
print(df.info())
print(df.describe())

# Exploratory Data Analysis
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
plt.show() # shows relationships betwen all variables of the dataset.. strong correlation between duration of membership and yearly amount spent 

# sns linear model plot 
sns.lmplot(x='Length of Membership',
           y='Yearly Amount Spent',
           data = df,
           scatter_kws={'alpha':0.3})
plt.show()

# split data into training and testing set. Target variable is Yearly amount spent what we want to predict. 
# TODO there is issues indexign dataframes 
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']] # Capital matrix
y = df['Yearly Amount Spent'] #Single variable 

# split with a 70/30 split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Multiple coefficient for each varriable
cdf = pd.DataFrame(lm.coef_, X.columns, columns=['Coef'])
print(cdf)

# Predictions -> what does the algo think the yearly amount spent will be based on the columns given
predictions = lm.predict(X_test)

# Plot these values against the actual values to see how close they are.. perfect would be a 1x1 straight line
sns.scatterplot(x=predictions, y=y_test)
plt.xlabel("Predictions")
plt.title("Evaluation of Linear Regression Model")
plt.show()

print("Mean Absolute Error: ", mean_absolute_error(y_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_test, predictions))
print("RMSE: ", math.sqrt(mean_squared_error(y_test, predictions)))

# residuals actual values - predictions, test for nomality which is assumed by linear models
residuals = y_test - predictions 
sns.displot(residuals, bins=15, kde=True)
plt.show()
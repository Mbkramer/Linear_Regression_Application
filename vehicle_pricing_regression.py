# linear regression practice with vehicle datasets to predict pricing
# Date: 11/08/2024  
# Description: This is independent, but informed from recent guided practice projects 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import pylab
import scipy.stats as stats

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import datasets

df = pd.read_csv('Datasets/CAR DETAILS CLEANED FOR REGRESSION ANALYSIS.csv')

print(df.head())
print(df.info())
print(df.describe())

#exploratory data analysis
# sns linear model plot 
sns.lmplot(x='year',
           y='selling_price',
           data = df,
           scatter_kws={'alpha':0.3})
plt.show()

sns.lmplot(x='km_driven',
           y='selling_price',
           data = df,
           scatter_kws={'alpha':0.3})
plt.show()

sns.lmplot(x='Number of Owners',
           y='selling_price',
           data = df,
           scatter_kws={'alpha':0.3})
plt.show()

# split data into training and testing set. Target variable is Yearly amount spent what we want to predict.  
X = df[['year', 'km_driven', 'Automatic', 'Number of Owners']] # Capital matrix
y = df['selling_price'] #Single variable 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

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
sns.displot(residuals, bins=50, kde=True)
plt.show()

# probability plot
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()
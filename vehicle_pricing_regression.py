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

og_df = pd.read_csv('Datasets/CAR DETAILS CLEANED FOR REGRESSION ANALYSIS.csv')

# split data into training and testing set. Target variable is Yearly amount spent what we want to predict.  
X_orig = og_df[['year', 'km_driven', 'Automatic', 'Number of Owners']] # Capital matrix
y_orig = og_df['selling_price'] #Single variable 

X_og_train, X_og_test, y_og_train, y_og_test = train_test_split(X_orig, y_orig, test_size=.25, random_state=42)

# Training the model
lm = LinearRegression()
lm.fit(X_og_train, y_og_train)

# Multiple coefficient for each varriable
cog_df = pd.DataFrame(lm.coef_, X_orig.columns, columns=['Coef'])
print(cog_df)

# Predictions -> what does the algo think the yearly amount spent will be based on the columns given
predictions = lm.predict(X_og_test)

# Plot these values against the actual values to see how close they are.. perfect would be a 1x1 straight line
sns.scatterplot(x=predictions, y=y_og_test)
plt.xlabel("Predictions")
plt.title("Evaluation of Linear Regression Model")
plt.show()

print("Mean Absolute Error: ", mean_absolute_error(y_og_test, predictions))
print("Mean Squared Error: ", mean_squared_error(y_og_test, predictions))
print("RMSE: ", math.sqrt(mean_squared_error(y_og_test, predictions)))

# residuals actual values - predictions, test for nomality which is assumed by linear models
residuals = y_og_test - predictions 
sns.displot(residuals, bins=50, kde=True)
plt.show()

# probability plot
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()

##########################
## Now for enhanced data
##########################

enhanced_df = pd.read_csv('Datasets/ENHANCED_CAR_DETAILS.csv')

# split data into training and testing set. Target variable is Yearly amount spent what we want to predict.  
X_en = enhanced_df[['Year', 'Kilometer', 'Number of Owners', 'Engine (CC)', 'Length', 'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity']] # Capital matrix
y_en = enhanced_df['Price'] #Single variable 

X_en_train, X_en_test, y_en_train, y_en_test = train_test_split(X_en, y_en, test_size=.25, random_state=42)

# Training the model
lm = LinearRegression()
lm.fit(X_en_train, y_en_train)

# Multiple coefficient for each varriable
cog_df = pd.DataFrame(lm.coef_, X_en.columns, columns=['Coef'])
print(cog_df)

# Predictions -> what does the algo think the yearly amount spent will be based on the columns given
predictions_en = lm.predict(X_en_test)

# Plot these values against the actual values to see how close they are.. perfect would be a 1x1 straight line
sns.scatterplot(x=predictions_en, y=y_en_test)
plt.xlabel("Predictions")
plt.title("Evaluation of Linear Regression Model")
plt.show()

print("Mean Absolute Error: ", mean_absolute_error(y_en_test, predictions_en))
print("Mean Squared Error: ", mean_squared_error(y_en_test, predictions_en))
print("RMSE: ", math.sqrt(mean_squared_error(y_en_test, predictions_en)))

# residuals actual values - predictions, test for nomality which is assumed by linear models
residuals = y_en_test - predictions_en
sns.displot(residuals, bins=50, kde=True)
plt.show()

# probability plot
stats.probplot(residuals, dist="norm", plot=pylab)
pylab.show()

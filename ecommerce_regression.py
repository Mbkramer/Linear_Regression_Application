# Intro to linear regression practice 
# Date: 11/06/2024  
# Description: This was built coding along side a tutorial video and is not original work. 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the model
lm = LinearRegression()
lm.fit(X_train, Y_train)

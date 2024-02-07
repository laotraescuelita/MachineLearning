#Import librarias

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets 
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor 
from sklearn import metrics
from sklearn.datasets import fetch_california_housing



#Load data 
#df = sklearn.datasets.load_boston()
df = fetch_california_housing()
print(dir(df))
print(df.data.shape)
print(df.feature_names)
print(df.target)
print(df.target_names)

df_ = pd.DataFrame(df.data, columns=df.feature_names)
print(df_.head())
df_["price"] = df.target 
print(df_.head())

#Analize the new data frame
print(df_.shape)
print(df_.isnull().sum())
print(df_.describe())

#Understanding correlation between vectors. 
correlation = df_.corr()
plt.figure(figsize=(6,6))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={"size":8}, cmap="Blues")
plt.show()

#Splitting the df and a matrix and a vector. 
X = df_.drop(["price"], axis=1)
y = df_["price"]

#Train test split 
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1 )
print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape	)

#Training the model 
model = XGBRegressor()
model.fit(X_train, Y_train)

#Accuracy with training data 
y_hat_train = model.predict(X_train)
R2 = metrics.r2_score(Y_train, y_hat_train)
R2_ = metrics.mean_absolute_error(Y_train, y_hat_train)
print("Accuracy", R2, R2_)

plt.scatter(Y_train, y_hat_train)
plt.xlabel("Actual prices")
plt.ylabel("Predict prices")
plt.title("Actual price vs predicted")
plt.show()

#Accuracy with testing data 
y_hat_test = model.predict(x_test)
R2 = metrics.r2_score(y_test, y_hat_test)
R2_ = metrics.mean_absolute_error(y_test, y_hat_test)
print("Accuracy", R2, R2_)





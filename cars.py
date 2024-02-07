#Import libraries 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso 
from sklearn import metrics 

#Loading data 

df = pd.read_csv("C:/Users/erick/OneDrive/Escritorio/Erick/DataStructures and Algorithms/Machinelearning/cars.csv")
print(df.shape)
print(df.head())

#Preprocessing data 
print(df.isnull().sum() / df.shape[0])
print(df.describe())
print(df.info())
print(df.Fuel_Type.value_counts())
print(df.Seller_Type.value_counts())
print(df.Transmission.value_counts())
#print(df.Car_Name.value_counts())

#Encoding the categorical data 
df.replace({
	"Fuel_Type":{"Petrol":0, "Diesel":1, "CNG":2},
	"Seller_Type":{"Dealer":0, "Individual":1},
	"Transmission":{"Manual":0, "Automatic":1},
	},inplace=True)

print(df.head())

#Splitting the data 
X = df.drop(["Car_Name","Selling_Price"], axis=1)
y = df["Selling_Price"]

#Splitting data into train and test vectors
X_train, x_test, Y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3)
print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape)

# Train and test model 
model = LinearRegression()

model.fit(X_train, Y_train)
y_hat_train = model.predict(X_train)
R2 = metrics.r2_score(Y_train, y_hat_train)
print("r square error", R2)

plt.scatter(Y_train, y_hat_train)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


model.fit(x_test, y_test)
y_hat_test = model.predict(x_test)
R2 = metrics.r2_score(y_test, y_hat_test)
print("r square error", R2)

plt.scatter(y_test, y_hat_test)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


model = Lasso()

model.fit(X_train, Y_train)
y_hat_train = model.predict(X_train)
R2 = metrics.r2_score(Y_train, y_hat_train)
print("r square error", R2)

plt.scatter(Y_train, y_hat_train)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


model.fit(x_test, y_test)
y_hat_test = model.predict(x_test)
R2 = metrics.r2_score(y_test, y_hat_test)
print("r square error", R2)

plt.scatter(y_test, y_hat_test)
plt.xlabel("Actual price")
plt.ylabel("Predicted price")
plt.title("Actual Prices vs Predicted Prices")
plt.show()

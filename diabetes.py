#This is a clasification problem we have to targets 1 and 0. 

#Import libraries

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

#Load data 

df = pd.read_csv("C:/Users/erick/OneDrive/Escritorio/Erick/DataStructures and Algorithms/Machinelearning/diabetes.csv")
print(df.head() )
print(df.shape )
print(df.describe())
print(df.info())
print(df["Outcome"].value_counts())
print( df.groupby("Outcome").mean())

#Separate the data frame in one matrix of values and one vector as the objective. 
X = df.drop(columns="Outcome", axis=1)
y = df["Outcome"]

#New shapes 
print( X.shape )
print( y.shape )


#Standardization 

scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)
print( X_standard	)

#Separate the data frame in one matrix of values and one vector as the objective. 
X = X_standard
y = df["Outcome"]

#Train test split 
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1 )
print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape	)

#Training the model 
model = svm.SVC(kernel="linear")
model.fit(X_train, Y_train)

#Accuracy with training data 
y_hat_train = model.predict(X_train)
accuracy = accuracy_score(y_hat_train, Y_train)
print("Accuracy", accuracy)

#Accuracy with testing data 
y_hat_test = model.predict(x_test)
accuracy = accuracy_score(y_hat_test, y_test)
print("Accuracy", accuracy)

#Predicting

#0
a1 = np.asarray( [5,116,74,0,0,25.6,0.201,30] )
#1
#a1 = np.asarray( [6,148,72,35,0,33.6,0.627,50] )
print( a1 )
a1_reshape = a1.reshape(1,-1)
print( a1_reshape )
a1_standard = scaler.transform(a1_reshape)
print( a1_standard )
prediction = model.predict( a1_standard )
print( prediction )

if prediction[0] == 0:
	print("positive")
else:
	print("negative")







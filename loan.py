#Import libraries

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm 
from sklearn.metrics import accuracy_score

#load the data 
df = pd.read_csv("C:/Users/erick/OneDrive/Escritorio/Erick/DataStructures and Algorithms/Machinelearning/loan.csv")
print( type(df))
print(df.head())
print(df.shape)
print(df.describe())

#preprocessing tha df.
print(df.isnull().sum()/df.shape[0]*100)

#Eliminate all the missing values because the max percentage is 8 which means that doesnt alter the matrix significantly.
df = df.dropna()
print(df.isnull().sum())

#Replace the target vector Y/N for 1/0 
df.replace({"Loan_Status":{"Y":1,"N":0}}, inplace=True)
print(df.head())

#Dependents column modify. 
print( df["Dependents"].value_counts() ) 
df = df.replace(to_replace="3+",value=4)
print( df["Dependents"].value_counts() ) 


#Show the relation of one vector basen on a second one. 
sns.countplot(x="Education", hue="Loan_Status", data=df)
plt.show()
sns.countplot(x="Married", hue="Loan_Status", data=df)
plt.show()

#Convert categorical vectors into numerical. 
df.replace( {"Married":{"No":0,"Yes":1},"Gender":{"Male":1,"Female":0},"Self_Employed":{"No":0,"Yes":1},
	"Property_Area":{"Rural":0,"Semiurban":1,"Urban":2}, "Education":{"Graduate":1,"Not Graduate":0}}, inplace=True)

print(df.head())

#Separating the matrix and teh vector to predict. 
X = df.drop(columns=["Loan_ID","Loan_Status"], axis=1)
y = df["Loan_Status"]

#New shapes 
print( X.shape )
print( y.shape )

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


"""
#Predicting
#0
a1 = np.asarray( [1,0,0,1,0,5849,0,,360,1,2] )
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
"""
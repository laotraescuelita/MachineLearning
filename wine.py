#Import libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Loading data
df = pd.read_csv("C:/Users/erick/OneDrive/Escritorio/Erick/DataStructures and Algorithms/Machinelearning/wine.csv")

#Preprocessing data
print(df.shape)
print(df.head())
print( (df.isnull().sum()/df.shape[0]) * 100)
print(df.describe())

#Visualize some vectors 

#plot = plt.figure(figsize=(5,5))
sns.catplot(x="quality", data=df, kind="count")
plt.show()
sns.barplot(x="quality",y="volatile acidity", data=df)
plt.show()
sns.barplot(x="quality",y="citric acid", data=df)
plt.show()

correlation = df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={"size":8}, cmap="Blues")
plt.show()

#Separate the matrix and the vector
X = df.drop("quality",axis=1)
#Label terget vector
y = df["quality"].apply(lambda y: 1 if y>=7 else 0 )

#Splitting data into train and test vectors
X_train, x_test, Y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=3)
print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape)

# Train and test model 
model = RandomForestClassifier()
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
a1 = np.asarray( [6.6,0.52,0.08,2.4,0.07,13.0,26.0,0.9935799999999999,3.4,0.72,12.5] )
#a1 = np.asarray( [7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4] )
print( a1 )
a1_reshape = a1.reshape(1,-1)
print( a1_reshape )
prediction = model.predict( a1_reshape )
print( prediction )

if prediction[0] == 1:
	print("Good quality")
else:
	print("Not good quality")





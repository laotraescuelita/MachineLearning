#Collect data, data pre processing, train test split, apply machine learning model. 

#Import libraries 
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score 


#Data collection and data processing 

data = pd.read_csv("C:/Users/erick/OneDrive/Escritorio/Erick/DataStructures and Algorithms/Machinelearning/sonar.csv", header=None)
print( data.shape)
print( data.describe())
print( data[60].value_counts())
print( data.groupby(60).mean())

#Separate the data frame in one matrix of values and one vector as the objective. 
X = data.drop(columns=60, axis=1)
y = data[60]

#New shapes 
print( X.shape )
print( y.shape )

#Train test split 
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1 )
print(X_train.shape, x_test.shape, Y_train.shape, y_test.shape	)

#Training the model 
model = LogisticRegression()
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
#"R"
#a1 = np.asarray( [0.0200,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.0660,0.2273,0.3100,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.5550,0.6711,0.6415,0.7104,0.8080,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.0510,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.0180,0.0084,0.0090,0.0032] )
a1 = np.asarray( [0.0094,0.0333,0.0306,0.0376,0.1296,0.1795,0.1909,0.1692,0.1870,0.1725,0.2228,0.3106,0.4144,0.5157,0.5369,0.5107,0.6441,0.7326,0.8164,0.8856,0.9891,1.0000,0.8750,0.8631,0.9074,0.8674,0.7750,0.6600,0.5615,0.4016,0.2331,0.1164,0.1095,0.0431,0.0619,0.1956,0.2120,0.3242,0.4102,0.2939,0.1911,0.1702,0.1010,0.1512,0.1427,0.1097,0.1173,0.0972,0.0703,0.0281,0.0216,0.0153,0.0112,0.0241,0.0164,0.0055,0.0078,0.0055,0.0091,0.0067] )
print( a1 )
a1_reshape = a1.reshape(1,-1)
print( a1_reshape )
prediction = model.predict( a1_reshape )
print( prediction )

if prediction[0] == "R":
	print("Object is a rock")
else:
	print("Object is a mine")





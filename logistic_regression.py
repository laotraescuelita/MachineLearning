import numpy as np
import matplotlib.pyplot as plt

#To try algorithm with diabetes data.
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class LogisticRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = [] 
        self.cost = []
        self.predict = []

    # Sigmoid function (logistic function)
    def _sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X_train, y_train):
        
        #Initiate weigths and bias a long with a seed to reproduce and get the same results.
        np.random.seed(42)
        self.weights = np.random.randn(X_train.shape[1],1)
        self.bias = np.random.randn(X_train.shape[0],1)

        # Train the model using gradient descent
        for i in range(self.epochs):

            #Logist means matrix multiplication betwwen weights and inputs + bias
            logits = np.dot(X_train, self.weights) + self.bias
            #Probabilities means that we apply an activation function in this specific excamle is the sigmoid. 
            probabilities = self._sigmoid(logits) 
            #Errors is the diferrence betwenn the original data and teh predicted ones.            
            errors = probabilities - y_train

            #Cost function will tell us if the algorithm is reducing or not. 
            cost = -(1/X_train.shape[1]) * np.sum(y_train*np.log(probabilities) + (1-y_train)*np.log(1-probabilities) )
            
            #We have to derivate the weights and bias with respect of ?? to apply teh gradient descent.
            d_weights = 1 / X_train.shape[1] * X_train.T.dot(errors)
            d_bias = 1 / X_train.shape[1] * np.sum(errors)
            
            #Update the weights and bias
            self.weights -= learning_rate * d_weights
            self.bias -= learning_rate * d_bias

            self.cost.append(cost)            

        return self.weights, self.bias, self.cost

    def _predict(self, X_test, weights):
        # Make predictions on the test data
        logits_test = np.dot(X_test, weights)
        probabilities_test = self._sigmoid(logits_test)
        self.predict = (probabilities_test >= 0.5).astype(int)
        return self.predict

    def _plot_cost(self, cost):
        plt.plot(cost, np.arange(0,self.epochs))
        plt.show()

    def _plot(self, X_test, y_test, y_pred):   
        plt.scatter(X_test, y_test, color="blue")
        plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted')        
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title("Logistic Regression Model")
        plt.show()

    def _accuracy(self, y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_instances = len(y_true)
        accuracy_value = correct_predictions / total_instances
        return accuracy_value 


# Generate some random data for demonstration
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = (4 + 3 * X + np.random.randn(100, 1)) > 6  # Binary classification problem

#Add a bias term to the input features. 
#X = np.c_[ np.ones((X.shape[0],1)), X ]

#Split data in train and test. 
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]


"""
#oading data
df = pd.read_csv("C:/Users/erick/OneDrive/Escritorio/Erick/DataScience/Machinelearning/ml_chatgpt/diabetes.csv")
#Separate the data frame in one matrix of values and one vector as the objective. 
X = df.drop(columns="Outcome", axis=1)
y = df["Outcome"]
#Standardization 
scaler = StandardScaler()
scaler.fit(X)
X_standard = scaler.transform(X)
#Separate the data frame in one matrix of values and one vector as the objective. 
X = X_standard
y = df["Outcome"]
#Train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1 )
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape  )
"""

# Set hyperparameters
learning_rate = 0.002
epochs = 1000


logreg = LogisticRegression(learning_rate, epochs)
weights, bias, cost = logreg.fit(X_train, y_train)
logreg._plot_cost(cost)
y_pred = logreg._predict(X_test, weights)
logreg._plot(X_test, y_test, y_pred)
accuracy_value = logreg._accuracy(y_test.flatten(), y_pred.flatten())
print(f'Accuracy: {accuracy_value}')

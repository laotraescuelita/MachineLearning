import numpy as np 
import matplotlib.pyplot as plt

class LinearRegression:
	def __init__(self, learning_rate, epochs):
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.cost = []
		self.theta = []
		self.y_hat = []		

	def fit(self, X_train, y_train):
		
		#Seed to get same results in the next exercise
		np.random.seed(42)
		#Initialize weights randomly
		self.theta = np.random.randn(X_train.shape[1],1)

		#Train the model using gradient descent 
		for _ in range(self.epochs):
			#Predicted values after a matriz multiplciation X x weights
			self.y_hat = np.dot(X_train, self.theta)
			#Errors is the diference between predicted values and actual values
			errors = self.y_hat - y_train
			
			#Cost function to determine if it is going down 
			cost = ( 1/(2*X_train.shape[0]) * np.sum(np.square(errors)) )

			#Gradient descent, we need to have the derivative of weights con respect to ??
			d_theta = (1/len(X_train)) * X_train.T.dot( errors ) 
			#Update the weights and 
			self.theta -= self.learning_rate * d_theta
			
			self.cost.append( cost )

		return self.cost, self.theta

	def _predict(self, X_test, theta):
		# Make predictions on the test data
		predict =  np.dot(X_test, theta)
		return predict

	def _plot(self, X_test, y_test, y_pred):
		#Plot the data and the regression line
		plt.scatter(X_test[:,1], y_test, color="black")		
		plt.plot(X_test[:,1], y_pred, color="blue", linewidth=3)
		plt.xlabel("X")
		plt.ylabel("y")
		plt.title("Linear Regression Model")
		plt.show()

	def _plot_cost(self, cost):
		plt.plot(cost, np.arange(0,self.epochs))
		plt.show()

	
	def r_squared(self, y_true, y_pred):
	    # Calculate the mean of the true values
	    mean_y = np.mean(y_true)

	    # Calculate the total sum of squares
	    total_sum_squares = np.sum((y_true - mean_y) ** 2)

	    # Calculate the residual sum of squares
	    residual_sum_squares = np.sum((y_true - y_pred) ** 2)

	    # Calculate R-squared
	    r2 = 1 - (residual_sum_squares / total_sum_squares)

	    return r2


#Generate random data  
np.random.seed(42)
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.rand(100,1)

#Add a bias term to the input features. 
X = np.c_[ np.ones((X.shape[0],1)), X ]

#Split data in train and test. 
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

#Set hypermateres
learning_rate = 0.01
epochs = 1000

#Instiate
linreg = LinearRegression(learning_rate, epochs)

#Cost ans weights
cost, theta = linreg.fit(X_train, y_train) 

#y predicted
y_pred = linreg._predict(X_test, theta) 

#Plot y_true vs y_predicted
linreg._plot(X_test, y_test, y_pred)

#Plot cost versus iterations 
linreg._plot_cost(cost)

# Calculate R-squared for the test data
r2_value = linreg.r_squared(y_test, y_pred)
print(f'R-squared: {r2_value}')

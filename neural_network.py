import numpy as np
import matplotlib.pyplot as plt

# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(x):
    return np.maximum(0, x)

def tanh(z):
    return np.tanh(z)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True )

#Derivatives of activation function
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)

# Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)

class NeuralNetworks:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost = []

    def fit(self, X_train, y_train):
        #Initialize  weights
        m, n  = X_train.shape       
        p = 10
        q, r  = y_train.shape        
        
        w1 = np.random.randn(n, p)
        b1 = np.zeros((m,1))
        w2 = np.random.randn(p, r)
        b2 = np.zeros((q,1))

        # Train the neural network using gradient descent
        for epoch in range(epochs):
            # Forward propagation
            z1 = np.dot(X_train, w1 ) + b1 #hidden_layer_input            
            a1 = relu(z1) #hidden_layer_output            
            z2 = np.dot(a1, w2) + b2 #output_layer            
            y_pred = softmax(z2) #output_layer
            
            
            # cross entropy function
            epsilon = 1e-15  # Small constant to avoid log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
            cost = -np.sum(y_train * np.log(y_pred)) / q

            #Backward propagation            
            dz2 = y_pred - y_train    
            dw2 = a1.T.dot(dz2)
            db2 = np.sum(dz2, axis=1, keepdims=True)

            dz1 = dz2.dot(w2.T) * relu_derivative(a1)
            dw1 = X_train.T.dot(dz1)
            db1 = np.sum(dz2, axis=1, keepdims=True)
            
            # Update weights and biases
            #weights_output += learning_rate * hidden_layer_output.T.dot(output_delta)
            #biases_output += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
            #weights_hidden += learning_rate * X_train_bias.T.dot(hidden_layer_delta)
            #biases_hidden += learning_rate * np.sum(hidden_layer_delta, axis=0, keepdims=True)

            w1 -= learning_rate*dw1
            b1 -= learning_rate*db1
            w2 -= learning_rate*dw2
            b2 -= learning_rate*db2

            self.cost.append( cost )

            #Imprimir el costo 
            if (epoch % (self.epochs / 10) == 0):
                print("Cost after", epoch, "Iteration is:", cost)

        return self.cost, y_pred, w1, b1, w2, b2
    

    """
    def _predict(self, X_test, w1, b1, w2, b2):
        # Make predictions on the test data
        a1 = np.dot(X_test, w1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, w2) + b2
        y_pred = sigmoid(output_layer_test_input)
    return y_pred
    """

    def _plot_cost(self, cost):
        plt.plot(cost, np.arange(0, self.epochs))
        plt.show()

    def _plot(self, X_test, y_test, y_pred):
        plt.scatter(X_test, y_test, color="blue")
        plt.scatter(X_test, y_pred, color='red', marker='x', label='Predicted')        
        plt.xlabel("X")
        plt.ylabel("y")
        plt.title('Neural Network (From Scratch)')
        plt.show()


# Generate some random data for demonstration
X = 2 * np.random.rand(100, 2)
y = np.random.randint(0, 10, size=(100, 1))

# One-hot encode the labels
y_one_hot = np.eye(10)[y.flatten()]

# Split the data into training and testing sets
X_train, X_test = X[:80], X[80:]
y_train, y_test = y_one_hot[:80], y_one_hot[80:]

# Set hyperparameters
learning_rate = 0.01
epochs = 10000

nn = NeuralNetworks(learning_rate, epochs)
cost, y_pred, w1, b1, w2, b2 = nn.fit(X_train, y_train)
nn._plot_cost(cost)
#cost, y_pred, w1, b1, w2, b2 = nn.fit(X_test, y_test)
#nn._plot_cost(cost)
#nn._plot(X_test, y_test, y_pred)

"""
# Convert probabilities to binary predictions
y_pred_binary = (y_pred_test >= 0.5).astype(int)

"""
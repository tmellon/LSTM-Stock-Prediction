import numpy as np

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# relu
def relu(X):
    return np.maximum(0, X)

# derivative of tanh
def tanh_derivative(X):
    return 1 - X**2

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
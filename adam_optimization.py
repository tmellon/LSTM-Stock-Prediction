from numpy.lib import emath
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class AdamOptimizer:
    def __init__(self, learning_rate=0.05, beta1=0.90, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update_parameters(self, parameters, gradients):
        self.t += 1

        for param_name, gradient in gradients.items():
            # Initialize the first moment estimate (mean)
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(gradient)
            
            # Initialize the second raw moment estimate (uncentered variance)
            if param_name not in self.v:
                self.v[param_name] = np.zeros_like(gradient)

            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient

            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (gradient ** 2)

            # Bias correction
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            # Update parameters
            parameters[param_name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return parameters 
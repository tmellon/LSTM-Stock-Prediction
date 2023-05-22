import numpy as np

class AdamOptimizer:
    def __init__(self, hyperparams):
        self.learning_rate = hyperparams['learning_rate']
        self.beta1 = hyperparams['beta1']
        self.beta2 = hyperparams['beta2']
        self.epsilon = hyperparams['epsilon']
        self.s = {}
        self.v = {}
        self.t = 0

    def update_parameters(self, parameters, gradients):
        self.t += 1

        for param_name, gradient in gradients.items():
            # Initialize the first moment estimate (mean)
            if param_name not in self.s:
                self.s[param_name] = np.zeros_like(gradient)
            
            # Initialize the second raw moment estimate (uncentered variance)
            if param_name not in self.v:
                self.v[param_name] = np.zeros_like(gradient)

            # Update biased first moment estimate
            self.v[param_name] = self.beta1 * self.v[param_name] + (1 - self.beta1) * gradient

            # Update biased second raw moment estimate
            self.s[param_name] = self.beta2 * self.s[param_name] + (1 - self.beta2) * (gradient ** 2)

            # Bias correction
            s_hat = self.s[param_name] / (1 - self.beta2 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta1 ** self.t)

            parameters[param_name[1:]] -= self.learning_rate * (self.v[param_name] / (np.sqrt(self.s[param_name]) + self.epsilon))
        #print(parameters)
        return parameters
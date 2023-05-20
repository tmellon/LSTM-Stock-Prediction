from numpy.lib import emath
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error

class AdamOptimizer:
    def __init__(self, learning_rate=0.005, beta1=0.90, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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
            s_hat = self.s[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            # Update parameters
            parameters[param_name[1:]] -= self.learning_rate * (self.v[param_name] / (np.sqrt(self.s[param_name]) + self.epsilon))
        #print(parameters)
        return parameters
    
class LSTM_RNN():
    def __init__(self, input_size=6, hidden_size=64, batch_size=100, epochs=1000):
        self.adam_optimizer = AdamOptimizer()
        self.scalar = MinMaxScaler()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.parameters = self.initialize_params()
        self.train_data, self.test_data = self.load_data()

        self.lstm_cache = []
        self.output_cache = []
        self.activation_cache = []
        self.cell_activation_cache = []

        self.lstm_error_cache = []

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # relu
    def relu(self, X):
        return np.maximum(0, X)

    # derivative of tanh
    def tanh_derivative(self, X):
        return 1 - X**2

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    #softmax activation
    def softmax(self, X):
        exp_X = np.exp(X)
        exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
        exp_X = exp_X/exp_X_sum
        return exp_X

    #Define LSTM weights
    def initialize_params(self):

        f = np.random.uniform(0, 1, (self.input_size + self.hidden_size, self.hidden_size)) # forget gate weights
        i = np.random.uniform(0, 1, (self.input_size + self.hidden_size, self.hidden_size)) # input gate weights
        o = np.random.uniform(0, 1, (self.input_size + self.hidden_size, self.hidden_size)) # output gate weights
        g = np.random.uniform(0, 1, (self.input_size + self.hidden_size, self.hidden_size)) # cell weights
        h = np.random.uniform(0, 1, (self.hidden_size, self.input_size))# hidden weights

        parameters = dict()
        parameters['f_weight'] = f
        parameters['i_weight'] = i
        parameters['g_weight'] = g
        parameters['o_weight'] = o
        parameters['h_weight'] = h

        #parameters['forget_bias'] = f
        #parameters['input_bias'] = i
        #parameters['gate_bias'] = g
        #parameters['output_bias'] = o
        #parameters['hidden_bias'] = h

        return parameters

    def clear_caches(self):
        self.lstm_cache.clear()
        self.output_cache.clear()
        self.activation_cache.clear()
        self.cell_activation_cache.clear()

        self.lstm_error_cache.clear()
        return 0


    def load_data(self):
        stock_data = yf.download('SPY', start='2018-01-01', end='2023-05-01')
        stock_data.head()

        plt.figure(figsize=(15, 8))
        plt.title('Stock Prices History')
        plt.plot(stock_data['Close'])
        plt.xlabel('Date')
        plt.ylabel('Prices ($)')
        plt.savefig('Original.png')

        stock_data_np = stock_data.to_numpy()

        #normalize data
        self.scalar.fit(stock_data_np)

        train_size = int(len(stock_data_np) * 0.8)

        train_data = self.scalar.transform(stock_data_np[0:train_size,:])
        test_data = self.scalar.transform(stock_data_np[train_size:len(stock_data_np),:])

        return train_data, test_data

    def batchify_data(self, data, batch_size):
        num_batches = len(data) // batch_size
        batches = []

        for data_group in range(num_batches):
            start = data_group * batch_size
            end = start + batch_size

            batch_data = data[start:end]

            if (len(batch_data) != batch_size):
                break

            batches.append(batch_data)
        return np.array(batches)

    def lstm_cell(self, batch_data, h_prev, c_prev):
        parameters = self.parameters

        concat_data = np.concatenate([batch_data, h_prev], axis=1)
        f_t = self.sigmoid(np.matmul(concat_data, parameters['f_weight']))
        i_t = self.sigmoid(np.matmul(concat_data, parameters['i_weight']))
        o_t = self.sigmoid(np.matmul(concat_data, parameters['o_weight']))
        g_t = np.tanh(np.matmul(concat_data, parameters['g_weight']))

        cell_activation = f_t * c_prev + i_t * g_t
        activation = o_t * np.tanh(cell_activation)

        state = dict()
        state['f_activation'] = f_t
        state['i_activation'] = i_t
        state['o_activation'] = o_t
        state['g_activation'] = g_t
        return state, activation, cell_activation

    def output_cell(self, activation_matrix):
        #get hidden to output parameters
        W_h = self.parameters['h_weight']
        
        #get outputs 
        output_matrix = np.matmul(activation_matrix, W_h)
        output_matrix = self.sigmoid(output_matrix)
        
        return output_matrix

    def forward_prop(self, batches):

        # define the initial states
        prev_activation = np.zeros((self.batch_size, self.hidden_size)) # (batch size, features)
        prev_cell_activation = np.zeros((self.batch_size, self.hidden_size)) # (batch size, features)

        for batch_data in batches:
            state, activation, cell_activation = self.lstm_cell(batch_data, prev_activation, prev_cell_activation)
            self.lstm_cache.append(state)
            self.activation_cache.append(activation)
            self.cell_activation_cache.append(cell_activation)

            prev_activation = activation
            prev_cell_activation = cell_activation
            
            #output cell
            output = self.output_cell(activation)
            self.output_cache.append(output)

        return 0

    #backpropagation
    def backward_prop(self, batch_data):
        parameters = self.parameters

        #calculate output errors 
        output_error_cache, activation_error_cache = self.calculate_output_error(batch_data)
        
        #to store lstm error for each time step
        
        # next activation error 
        # next cell error  
        #for last cell will be zero
        next_activation_error = np.zeros(activation_error_cache[0].shape)
        next_cell_error = np.zeros(activation_error_cache[0].shape)
        
        #calculate all lstm cell errors (going from last time-step to the first time step)
        for i in reversed(range(len(self.lstm_cache))):
            #calculate the lstm errors for this time step 't'
            prev_activation_error, prev_cell_error, lstm_error, lstm_error = self.calculate_lstm_error(activation_error_cache[i], next_activation_error, next_cell_error, self.lstm_cache[i], self.cell_activation_cache[i], self.cell_activation_cache[i - 1])
            
            #store the lstm error in dict
            self.lstm_error_cache.append(lstm_error)
            
            #update the next activation error and next cell error for previous cell
            next_activation_error = prev_activation_error
            next_cell_error = prev_cell_error
        
        #calculate output cell derivatives
        derivatives = dict()
        derivatives['dh_weight'] = self.calculate_output_derivatives(output_error_cache, self.activation_cache)
        
        #calculate lstm cell derivatives for each time step and store in lstm_derivatives dict
        lstm_derivatives = []
        for i in range(len(self.lstm_error_cache)):
            lstm_derivatives.append(self.calculate_lstm_derivatives(batch_data, self.lstm_error_cache[i], self.activation_cache[i]))
        
        #initialize the derivatives to zeros 
        derivatives['df_weight'] = np.zeros(parameters['f_weight'].shape)
        derivatives['di_weight'] = np.zeros(parameters['i_weight'].shape)
        derivatives['do_weight'] = np.zeros(parameters['o_weight'].shape)
        derivatives['dg_weight'] = np.zeros(parameters['g_weight'].shape)
        
        #sum up the derivatives for each time step
        for i in range(len(self.lstm_error_cache)):
            derivatives['df_weight'] += lstm_derivatives[i]['df_weight']
            derivatives['di_weight'] += lstm_derivatives[i]['di_weight']
            derivatives['do_weight'] += lstm_derivatives[i]['do_weight']
            derivatives['dg_weight'] += lstm_derivatives[i]['dg_weight']
        
        return derivatives
    
    def train(self):
        #to store the Loss, Perplexity and Accuracy for each batch
        J = []
        P = []
        A = []
        
        batch_data = self.batchify_data(self.train_data, self.batch_size)
        
        for step in range(self.epochs):
            index = step % len(batch_data)
            batches = batch_data[index]
            
            #forward propagation
            self.forward_prop([batches])
            
            #backward propagation
            derivatives = self.backward_prop(batches)
            
            #update the parameters
            self.parameters = self.adam_optimizer.update_parameters(self.parameters, derivatives)

            self.clear_caches()
        return 0
    
    def calculate_output_error(self, batch_data):
        output_error_cache = []
        activation_error_cache = []

        hidden_weight = self.parameters['h_weight']

        for i in range(len(self.output_cache)):
            actual = batch_data[i]
            pred = self.output_cache[i]

            output_error = pred - actual

            activation_error = np.matmul(output_error, hidden_weight.T)

            output_error_cache.append(output_error)
            activation_error_cache.append(activation_error)

        return output_error_cache, activation_error_cache

    def calculate_lstm_error(self, activation_error, next_activation_error, next_cell_error, lstm_cache, cell_activation, prev_cell_activation):
        #activation error =  error coming from output cell and error coming from the next lstm cell
        act_error = activation_error + next_activation_error
        
        #output gate error
        o_activation = lstm_cache['o_activation']
        o_error = np.multiply(act_error, np.tanh(cell_activation))
        o_error = np.multiply(np.multiply(o_error, o_activation), 1 - o_error)
        
        #cell activation error
        c_error = np.multiply(act_error, o_error)
        c_error = np.multiply(c_error, self.tanh_derivative(np.tanh(cell_activation)))
        #error also coming from next lstm cell 
        c_error += next_cell_error
        
        #input gate error
        i_activation = lstm_cache['i_activation']
        g_activation = lstm_cache['g_activation']

        i_error = np.multiply(c_error, g_activation)
        i_error = np.multiply(np.multiply(i_error, i_activation), 1 - i_activation)
        
        #gate gate error
        g_error = np.multiply(c_error, i_activation)
        g_error = np.multiply(g_error, self.tanh_derivative(g_activation))
        
        #forget gate error
        f_activation = lstm_cache['f_activation']
        f_error = np.multiply(c_error , prev_cell_activation)
        f_error = np.multiply(np.multiply(f_error, f_activation), 1 - f_activation)
        
        #prev cell error
        prev_cell_error = np.multiply(c_error, f_activation)
        
        #get parameters
        parameters = self.parameters
        f_weight = parameters['f_weight']
        i_weight = parameters['i_weight']
        g_weight = parameters['g_weight']
        o_weight = parameters['o_weight']
        
        #embedding + hidden activation error
        input_activation_error = np.matmul(f_error, f_weight.T)
        input_activation_error += np.matmul(i_error, i_weight.T)
        input_activation_error += np.matmul(o_error, o_weight.T)
        input_activation_error += np.matmul(g_error, g_weight.T)
        
        input_hidden_units = f_weight.shape[0]
        hidden_units = f_weight.shape[1]
        input_units = input_hidden_units - hidden_units
        
        #prev activation error
        prev_activation_error = input_activation_error[:,input_units:]
        
        #input error (embedding error)
        input_error = input_activation_error[:,:input_units]
        
        #store lstm error
        lstm_error = dict()
        lstm_error['f_error'] = f_error
        lstm_error['i_error'] = i_error
        lstm_error['o_error'] = o_error
        lstm_error['g_error'] = g_error
        
        return prev_activation_error, prev_cell_error, input_error, lstm_error
    
    #calculate output cell derivatives
    def calculate_output_derivatives(self, output_error_cache, activation_cache):
        #to store the sum of derivatives from each time step
        dh_weight = np.zeros(self.parameters['h_weight'].shape)
                
        #loop through the time steps 
        for t in range(len(output_error_cache)):
            #get output error
            output_error = output_error_cache[t]
            
            #get input activation
            activation = activation_cache[t]
            
            #cal derivative and summing up!
            dh_weight += np.matmul(activation.T,output_error) / self.batch_size
            
        return dh_weight
    
    #calculate derivatives for single lstm cell
    def calculate_lstm_derivatives(self, batch_data, lstm_error, activation_matrix):
        #get error for single time step
        ef = lstm_error['f_error']
        ei = lstm_error['i_error']
        eo = lstm_error['o_error']
        eg = lstm_error['g_error']
        
        #get input activations for this time step
        concat_matrix = np.concatenate((batch_data, activation_matrix), axis=1)
        
        batch_size = batch_data.shape[0]
        
        #cal derivatives for this time step
        df_weight = np.matmul(concat_matrix.T,ef)/batch_size
        di_weight = np.matmul(concat_matrix.T,ei)/batch_size
        do_weight = np.matmul(concat_matrix.T,eo)/batch_size
        dg_weight = np.matmul(concat_matrix.T,eg)/batch_size
        
        #store the derivatives for this time step in dict
        derivatives = dict()
        derivatives['df_weight'] = df_weight
        derivatives['di_weight'] = di_weight
        derivatives['do_weight'] = do_weight
        derivatives['dg_weight'] = dg_weight
        
        return derivatives

    def predict(self, batch_size):
        predictions = []  # Store the predicted values

        prev_activation = np.zeros((batch_size, self.hidden_size))  # Initialize previous activation
        prev_cell_activation = np.zeros((batch_size, self.hidden_size))  # Initialize previous cell activation

        batches = self.batchify_data(self.test_data, batch_size)
        batch_data = batches[0]

        for _ in range(len(batches)):
            _, activation, cell_activation = self.lstm_cell(batch_data, prev_activation, prev_cell_activation)

            prev_activation = activation
            prev_cell_activation = cell_activation
            
            # Output cell
            prediction = self.output_cell(activation)
            predictions.append(prediction)

            batch_data = prediction

        # Concatenate and reshape the predictions
        predictions = np.concatenate(predictions)

        normalized_pred = self.scalar.inverse_transform(predictions)

        return predictions, normalized_pred

    def calculate_direction_accuracy(self, predictions, window_size):
        correct_predictions = 0
        total_predictions = 0

        for i in range(window_size, len(predictions)):
            # Get the current prediction and actual movement
            current_prediction = predictions[i]
            previous_prediction = predictions[i - window_size]
            current_actual = self.test_data[i]
            previous_actual = self.test_data[i - window_size]

            # Compare the signs of the current prediction and actual movement
            prediction_direction = np.sign(current_prediction - previous_prediction)
            actual_direction = np.sign(current_actual - previous_actual)

            # Check if the signs match
            if prediction_direction.all == actual_direction.all:
                correct_predictions += 1

            total_predictions += 1

        direction_accuracy = (correct_predictions / total_predictions) * 100

        return direction_accuracy

    def visualize_results(self, predictions):
        df = pd.DataFrame(predictions, columns =['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        plt.figure(figsize=(15, 8))
        plt.title('Stock Prices History')
        plt.plot(df['Close'])
        plt.xlabel('Date')
        plt.ylabel('Prices ($)')
        plt.savefig('guess2.png')
        return 0
    
    def evaluate(self, predictions):
        y_true = self.scalar.inverse_transform(self.test_data)[:, 3]
        mae = mean_absolute_error(predictions, y_true)
        r2 = r2_score(predictions, y_true)
        return mae, r2
    
    import tensorflow as tf

    def run_lstm_with_adam_optimizer(x_train, y_train, num_units, num_layers, num_epochs, batch_size):
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(units=64,
                                    return_sequences=True,
                                    input_shape=(x_train.shape[1], 1)))
        model.add(keras.layers.LSTM(units=64))
        model.add(keras.layers.Dense(32))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))
        model.summary

        return model

    # Extract the actual values from the test data

    #print(predictions)

    # Denormalize the predictions and test data
    #predictions = scalar.inverse_transform(predictions)
    #test_data = scalar.inverse_transform(test_data)

    #actual_values = test_data[:, 4]  # Assuming the last column contains the target variable (stock prices)
    #window_size = 10

    # Evaluate the predictions
    # Calculate the direction accuracy
    #direction_acc = calculate_direction_accuracy(predictions[:, 4], actual_values, window_size)

    # Print the direction accuracy
    #print("Direction Accuracy:", direction_acc)

def main():
    lstm = LSTM_RNN()
    lstm.train()
    predictions, normalized_predictions = lstm.predict(10)
    #print(normalized_predictions[:100])
    #print(lstm.calculate_direction_accuracy(predictions, 10))
    #r2, mae = lstm.evaluate(normalized_predictions[:, 3])
    #print(normalized_predictions)
    lstm.visualize_results(normalized_predictions)
    #print('mae', mae, 'r2', r2)
    

if __name__ == "__main__":
    main()
from numpy.lib import emath
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from adam_optimizer import AdamOptimizer
    
class LSTM:
    def __init__(self, hyperparams, optimizer, data):
        self.adam_optimizer = optimizer
        self.scalar = MinMaxScaler()

        self.input_size = hyperparams['input_size']
        self.hidden_size = hyperparams['hidden_size']
        self.batch_size = hyperparams['batch_size']
        self.epochs = hyperparams['epochs']

        self.parameters = self.initialize_params()
        self.train_data, self.test_data = self.load_data(data['ticker'], data['start_date'], data['end_date'], data['split'])

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
        exp_X_sum = np.sum(exp_X,axis=1).reshape(-1, 1)
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

        fb = np.random.uniform(0, 1, (self.batch_size, self.hidden_size)) # forget gate weights
        ib = np.random.uniform(0, 1, (self.batch_size, self.hidden_size)) # input gate weights
        ob = np.random.uniform(0, 1, (self.batch_size, self.hidden_size)) # output gate weights
        gb = np.random.uniform(0, 1, (self.batch_size, self.hidden_size)) # cell weights
        parameters['f_bias'] = fb
        parameters['i_bias'] = ib
        parameters['g_bias'] = gb
        parameters['o_bias'] = ob

        return parameters
    
    def load_data(self, ticker, start_date, end_date, split):
        stock_data = yf.download(ticker, start=start_date, end=end_date)
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
        stock_data_np = self.scalar.transform(stock_data_np)

        # Get the training size
        train_size = int(len(stock_data_np) * split)
        test_size = len(stock_data_np) - train_size

        # Make the training size divisible by the batch_size
        ones1 = train_size % 10
        ones_diff1 = ones1 - (train_size % self.batch_size)
        if(ones_diff1 != 0):
            train_size += ones_diff1

        ones2 = train_size % 10
        ones_diff2 = test_size % self.batch_size
        if(ones_diff2 != 0):
            test_size -= ones_diff2

        train_data = stock_data_np[0:train_size,:]
        test_data = stock_data_np[train_size:train_size + test_size,:]

        return train_data, test_data

    def clear_caches(self):
        # Clear caches for next pass through
        self.lstm_cache.clear()
        self.output_cache.clear()
        self.activation_cache.clear()
        self.cell_activation_cache.clear()

        self.lstm_error_cache.clear()
        return 0

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

        cell_activation = np.multiply(f_t, c_prev) + np.multiply(i_t, g_t)
        activation = np.multiply(o_t, np.tanh(cell_activation))
        
        # Store lstm cell activations
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
        #output_matrix = self.sigmoid(output_matrix)
        
        return output_matrix

    def forward_prop(self, batches, batch_size):

        # define the initial states
        prev_activation = np.zeros((batch_size, self.hidden_size)) # (batch size, features)
        prev_cell_activation = np.zeros((batch_size, self.hidden_size)) # (batch size, features)

        for batch_data in batches:
            state, activation, cell_activation = self.lstm_cell(batch_data, prev_activation, prev_cell_activation)

            # Update caches
            self.lstm_cache.append(state)
            self.activation_cache.append(activation)
            self.cell_activation_cache.append(cell_activation)

            # Set values for previous activations to current cell values
            prev_activation = activation
            prev_cell_activation = cell_activation

            # Get output
            output = self.output_cell(activation)
            self.output_cache.append(output)
        return 0

    #backpropagation
    def backward_prop(self, batch_data):
        # Stores derivatives of weights
        derivatives = dict()
        derivatives['do_weight'] = np.zeros(self.parameters['o_weight'].shape)
        derivatives['do_bias'] = np.zeros(self.parameters['o_bias'].shape)
        derivatives['df_weight'] = np.zeros(self.parameters['f_weight'].shape)
        derivatives['df_bias'] = np.zeros(self.parameters['f_bias'].shape)
        derivatives['di_weight'] = np.zeros(self.parameters['i_weight'].shape)
        derivatives['di_bias'] = np.zeros(self.parameters['i_bias'].shape)
        derivatives['dg_weight'] = np.zeros(self.parameters['g_weight'].shape)
        derivatives['dg_bias'] = np.zeros(self.parameters['g_bias'].shape)

        #calculate output errors 
        output_error_cache, activation_error_cache = self.calculate_output_error(batch_data)
                
        next_activation_error = np.zeros(activation_error_cache[0].shape)
        next_cell_error = np.zeros(activation_error_cache[0].shape)
        
        #calculate all lstm cell errors (going from last time-step to the first time step)
        for i in reversed(range(len(self.lstm_cache))):
            lstm_cache = self.lstm_cache[i]
            f_activation = lstm_cache['f_activation']
            g_activation = lstm_cache['g_activation']
            o_activation = lstm_cache['o_activation']
            i_activation = lstm_cache['i_activation']
            
            concat_matrix = np.concatenate((batch_data, self.activation_cache[i]), axis=1)

            hidden_error = activation_error_cache[i] + next_activation_error

            cell_activation = self.cell_activation_cache[i]
            prev_cell_activation = self.cell_activation_cache[i - 1]

            # Output Gate Weights and Biases Errors
            o_error = np.tanh(cell_activation) * hidden_error * self.sigmoid_derivative(o_activation)
            derivatives['do_weight'] += np.matmul(concat_matrix.T, o_error)
            derivatives['do_bias'] += o_error

            # Cell State Error
            dc_error = self.tanh_derivative(np.tanh(cell_activation)) * o_activation * hidden_error + next_cell_error

            # Forget Gate Weights and Biases Errors
            f_error = dc_error * prev_cell_activation * self.sigmoid_derivative(f_activation)
            derivatives['df_weight'] += np.matmul(concat_matrix.T, f_error) / self.batch_size
            derivatives['df_bias'] += f_error / self.batch_size

            # Input Gate Weights and Biases Errors
            i_error = dc_error * g_activation * self.sigmoid_derivative(i_activation)
            derivatives['di_weight'] += np.matmul(concat_matrix.T, i_error) / self.batch_size
            derivatives['di_bias'] += i_error / self.batch_size
            
            # Candidate Gate Weights and Biases Errors
            g_error = dc_error * i_activation * self.tanh_derivative(g_activation)
            derivatives['dg_weight'] += np.matmul(concat_matrix.T, g_error) / self.batch_size
            derivatives['dg_bias'] += g_error / self.batch_size

            # Hidden activation error
            input_activation_error = np.matmul(f_error, self.parameters['f_weight'].T)
            input_activation_error += np.matmul(i_error, self.parameters['i_weight'].T)
            input_activation_error += np.matmul(o_error, self.parameters['o_weight'].T)
            input_activation_error += np.matmul(g_error, self.parameters['g_weight'].T)
            
            set_shape = self.parameters['f_weight']
            input_hidden_units = set_shape.shape[0]
            hidden_units = set_shape.shape[1]
            input_units = input_hidden_units - hidden_units
            
            # Prev activation error
            next_activation_error = input_activation_error[:,input_units:]

        #calculate output cell derivatives
        derivatives['dh_weight'] = self.calculate_output_derivatives(output_error_cache, self.activation_cache)
        
        return derivatives
    
    def train(self):
        # Create batches from training data
        batch_data = self.batchify_data(self.train_data, self.batch_size)

        for step in range(self.epochs):
            for batches in batch_data:                
                # Forward propagation
                self.forward_prop([batches], self.batch_size)

                # Backward propagation
                derivatives = self.backward_prop(batches)

                # Update the parameters
                self.parameters = self.adam_optimizer.update_parameters(self.parameters, derivatives)

                self.clear_caches()
        return
    
    def calculate_output_error(self, batch_data):
        output_error_cache = []
        activation_error_cache = []

        hidden_weight = self.parameters['h_weight']

        for i in range(len(self.output_cache)):
            actual = batch_data[i]
            pred = self.output_cache[i]

            output_error = pred - actual
            #output_error = mean_squared_error(actual, pred)
            #print(output_error)
            activation_error = np.matmul(output_error, hidden_weight.T)

            output_error_cache.append(output_error)
            activation_error_cache.append(activation_error)

        return output_error_cache, activation_error_cache
    
    def calculate_output_derivatives(self, output_error_cache, activation_cache):
        # To store the sum of derivatives from each time step
        dh_weight = np.zeros(self.parameters['h_weight'].shape)
                
        for t in range(len(output_error_cache)):
            # Get output error
            output_error = output_error_cache[t]
            
            # Get input activation
            activation = activation_cache[t]
            
            # Calculate derivative
            dh_weight += np.matmul(activation.T, output_error) / self.batch_size
            
        return dh_weight

    def predict(self, batch_size):
        predictions = []    # Stores the predicted values

        prev_activation = np.zeros((batch_size, self.hidden_size))  # Initialize previous activation
        prev_cell_activation = np.zeros((batch_size, self.hidden_size))  # Initialize previous cell activation
        
        # Prime LSTM model for predictions
        data = self.train_data[:]
        batch_data = self.batchify_data(data, batch_size)
        
        self.forward_prop(batch_data, batch_size)
        prev_activation = self.activation_cache[-1]
        #print(type(prev_activation))
        prev_cell_activation = self.cell_activation_cache[-1]
        #print(prev_cell_activation)
        batch_data = self.output_cache[-1]

        # Predict each batch
        for i in range(len(self.test_data)):
            state, activation, cell_activation = self.lstm_cell(batch_data, prev_activation, prev_cell_activation)
            prev_activation = activation
            prev_cell_activation = cell_activation
            
            # Output cell
            prediction = self.output_cell(activation)
            predictions.append(prediction)


            batch_data = prediction

        
        #print(predictions)
        #predictions = self.scalar.fit_transform(predictions)
        #normalized_pred = self.scalar.inverse_transform(predictions)
        predictions = np.asarray(predictions)
        predictions = predictions[:, 0, :]

        return predictions, #normalized_pred

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
            if prediction_direction == actual_direction:
                correct_predictions += 1

            total_predictions += 1

        direction_accuracy = (correct_predictions / total_predictions) * 100

        return direction_accuracy

    def visualize_results(self, predictions):
        #actual = self.scalar.inverse_transform(self.test_data)
        predictions[0]
        df = pd.DataFrame(predictions[0], columns =['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        plt.figure(figsize=(15, 8))
        plt.title('Stock Prices Predictions')
        plt.plot(df['Close'])
        plt.xlabel('Date')
        plt.ylabel('Prices ($)')
        plt.savefig('predicted.png')
        '''
        df = pd.DataFrame(actual, columns =['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        plt.figure(figsize=(15, 8))
        plt.title('Stock Prices Actual')
        plt.plot(df['Close'])
        plt.xlabel('Date')
        plt.ylabel('Prices ($)')
        plt.savefig('actual.png')
        '''
        return 0
    
    # Change to MSE
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

def main():
    adam_optimizer = AdamOptimizer({
        'learning_rate':    0.05, 
        'beta1':            0.90, 
        'beta2':            0.999, 
        'epsilon':          1e-8
    })

    data = {
        'ticker':       'SPY',
        'start_date':   '2018-01-01',
        'end_date':     '2023-01-01',
        'split':        0.8
    }

    lstm = LSTM({
        'input_size':       6, 
        'hidden_size':      64, 
        'batch_size':       10, 
        'epochs':           100
    }, adam_optimizer, data)
    lstm.train()
    predictions = lstm.predict(10)
    #print(normalized_predictions[:100])
    #print(lstm.calculate_direction_accuracy(predictions, 10))
    #r2, mae = lstm.evaluate(normalized_predictions[:, 3])
    #print(normalized_predictions)
    lstm.visualize_results(predictions)
    #print('mae', mae, 'r2', r2)
    

if __name__ == "__main__":
    main()
from numpy.lib import emath
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from activation_functions import *
from adam_optimization import AdamOptimizer

INPUT_SIZE = 6
HIDDEN_SIZE = 64
BATCH_SIZE = 755
EPOCHS = 1000

# Define LSTM weights

def initialize_params():
    W_f = np.random.uniform(0, 1, (INPUT_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)) # forget gate weights
    W_i = np.random.uniform(0, 1, (INPUT_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)) # input gate weights
    W_o = np.random.uniform(0, 1, (INPUT_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)) # output gate weights
    W_g = np.random.uniform(0, 1, (INPUT_SIZE + HIDDEN_SIZE, HIDDEN_SIZE)) # cell weights
    W_oc = np.random.uniform(0, 1, (HIDDEN_SIZE, INPUT_SIZE))# output cell weights

    parameters = dict()
    parameters['forget_weight'] = W_f
    parameters['input_weight'] = W_i
    parameters['gate_weight'] = W_g
    parameters['output_weight'] = W_o
    parameters['output_cell_weight'] = W_oc

    return parameters

# initialize the optimizer variables
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8

def load_data():
    stock_data = yf.download('SPY', start='2018-01-01', end='2023-01-01')
    stock_data.head()

    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History')
    plt.plot(stock_data['Close'])
    plt.xlabel('Date')
    plt.ylabel('Prices ($)')
    plt.savefig('Original.png')

    stock_data_np = np.array(stock_data)

    #normalize data
    scalar = MinMaxScaler()
    scalar.fit(stock_data_np)

    train_size = int(len(stock_data_np) * 0.6)

    train_data = scalar.transform(stock_data_np[0:train_size,:])
    test_data = scalar.transform(stock_data_np[train_size:len(stock_data_np),:])


    return train_data, test_data, scalar

def batchify_data(data):
    num_batches = len(data) // BATCH_SIZE
    batches = []

    for batch_index in range(num_batches):
        start = batch_index * BATCH_SIZE
        end = start + BATCH_SIZE

        batch_data = data[start:end]

        if (len(batch_data) != BATCH_SIZE):
            break

        batches.append(batch_data)

    return batches

def lstm_cell(parameters, batch_data, h_prev, c_prev):
    W_f = parameters['forget_weight']
    W_i = parameters['input_weight']
    W_o = parameters['output_weight']
    W_g = parameters['gate_weight']

    concat_data = np.concatenate([batch_data, h_prev], axis=1)
    f_t = sigmoid(np.matmul(concat_data, W_f))
    i_t = sigmoid(np.matmul(concat_data, W_i))
    o_t = sigmoid(np.matmul(concat_data, W_o))
    g_t = np.tanh(np.matmul(concat_data, W_g))

    cell_activation = f_t * c_prev + i_t * g_t
    activation = o_t * np.tanh(cell_activation)

    state = dict()
    state['f_activation'] = f_t
    state['i_activation'] = i_t
    state['o_activation'] = o_t
    state['g_activation'] = g_t
    return state, activation, cell_activation

def output_cell(parameters, activation_matrix):
    #get hidden to output parameters
    W_oc = parameters['output_cell_weight']
    
    #get outputs 
    output_matrix = np.matmul(activation_matrix, W_oc)
    output_matrix = relu(output_matrix)
    
    return output_matrix

def forward_prop(parameters, batches):
    #x = np.random.randn(1, INPUT_SIZE, 1) # (time steps, features, 1)

    # define the initial states
    prev_activation = np.zeros((BATCH_SIZE, HIDDEN_SIZE)) # (batch size, features)
    prev_cell_activation = np.zeros((BATCH_SIZE, HIDDEN_SIZE)) # (batch size, features)

    state_cache = []
    output_cache = []
    activation_cache = []
    cell_activation_cache = []

    for batch_data in batches:
        state, activation, cell_activation = lstm_cell(parameters, batch_data, prev_activation, prev_cell_activation)
        state_cache.append(state)
        activation_cache.append(activation)
        cell_activation_cache.append(cell_activation)

        prev_activation = activation
        prev_cell_activation = cell_activation
        
        #output cell
        output = output_cell(parameters, activation)
        output_cache.append(output)

    return state_cache, activation_cache, cell_activation_cache, output_cache

def backward_prop(batch_data, parameters, state_cache, activation_cache, cell_activation_cache, output_cache):
    dW_f = np.zeros_like(parameters['forget_weight'])
    dW_i = np.zeros_like(parameters['input_weight'])
    dW_o = np.zeros_like(parameters['output_weight'])
    dW_g = np.zeros_like(parameters['gate_weight'])
    dW_oc = np.zeros_like(parameters['output_cell_weight'])
    dh_next = np.zeros_like(activation_cache[0])
    dc_next = np.zeros_like(cell_activation_cache[0])

    for t in reversed(range(len(state_cache), 1)):
        state = state_cache[t]
        activation = activation_cache[t]
        cell_activation = cell_activation_cache[t]
        output = output_cache[t]

        # Compute the gradients
        do = output - batch_data[t]
        dW_oc += np.dot(activation.T, do)

        dh = np.dot(do, parameters['output_cell_weight'].T) + dh_next
        dc = dh * state['o_activation'] * (1 - np.tanh(cell_activation) ** 2) + dc_next

        dg = dc * state['i_activation'] * (1 - state['g_activation'] ** 2)
        dW_g += np.dot(np.concatenate([batch_data[t], activation_cache[t - 1]], axis=1).T, dg)

        di = dc * state['g_activation']
        dW_i += np.dot(np.concatenate([batch_data[t], activation_cache[t - 1]], axis=1).T, di)

        df = dc * cell_activation[t - 1]
        dW_f += np.dot(np.concatenate([batch_data[t], activation_cache[t - 1]], axis=1).T, df)

        dprev_c = dc * state['f_activation']

        # Compute the gradient for the previous activation and cell activation
        dprev_h = np.dot(df, parameters['forget_weight'].T)
        dprev_h += np.dot(di, parameters['input_weight'].T)
        dprev_h += np.dot(dg, parameters['gate_weight'].T)
        dprev_h += np.dot(do, parameters['output_weight'].T)

        # Store the gradients for the next iteration
        dh_next = dprev_h[:, -HIDDEN_SIZE:]
        dc_next = dprev_c

    # Clip the gradients to avoid exploding gradients
    for gradient in [dW_f, dW_i, dW_o, dW_g, dW_oc]:
        np.clip(gradient, -1, 1, out=gradient)

    gradients = {
        'forget_weight': dW_f,
        'input_weight': dW_i,
        'output_weight': dW_o,
        'gate_weight': dW_g,
        'output_cell_weight': dW_oc
    }

    return gradients

def train(train_data):
    adam_optimizer = AdamOptimizer()
    parameters = initialize_params()
    
    #to store the Loss, Perplexity and Accuracy for each batch
    J = []
    P = []
    A = []
    
    batch_data = batchify_data(train_data)
    
    for step in range(EPOCHS):
        index = step % len(batch_data)
        batches = [batch_data[index]]
        
        #forward propagation
        state_cache, activation_cache, cell_activation_cache, output_cache = forward_prop(parameters, batches)
        
        #calculate the loss, perplexity and accuracy
        #perplexity, loss, acc = cal_loss_accuracy(batches, output_cache)
        
        #backward propagation
        gradients = backward_prop(batches, parameters, state_cache, activation_cache, cell_activation_cache, output_cache)
        
        #update the parameters
        parameters = adam_optimizer.update_parameters(parameters, gradients)
        
        #J.append(loss)
        #P.append(perplexity)
        #A.append(acc)
        
        #print loss, accuracy and perplexity
        #print("For Single Batch :")
        #print('Step       = {}'.format(step))
        #print('Loss       = {}'.format(round(loss, 2)))
        #print('Perplexity = {}'.format(round(perplexity, 2)))
        #print('Accuracy   = {}'.format(round(acc*100, 2)))
        #print()
    return parameters, J, P, A

def predict(parameters, test_data):
    predictions = []  # Store the predicted values

    prev_activation = np.zeros((1, HIDDEN_SIZE))  # Initialize previous activation
    prev_cell_activation = np.zeros((1, HIDDEN_SIZE))  # Initialize previous cell activation

    batch_data = [test_data[0]]

    for _ in range(len(test_data)):
        _, activation, cell_activation = lstm_cell(parameters, batch_data, prev_activation, prev_cell_activation)

        prev_activation = activation
        prev_cell_activation = cell_activation
        
        # Output cell
        prediction = output_cell(parameters, activation)
        predictions.append(prediction)

        batch_data = prediction

    # Concatenate and reshape the predictions
    predictions = np.concatenate(predictions)

    return predictions



def calculate_direction_accuracy(predictions, actual_values, window_size):
    correct_predictions = 0
    total_predictions = 0

    for i in range(window_size, len(predictions)):
        # Get the current prediction and actual movement
        current_prediction = predictions[i]
        previous_prediction = predictions[i - window_size]
        current_actual = actual_values[i]
        previous_actual = actual_values[i - window_size]

        # Compare the signs of the current prediction and actual movement
        prediction_direction = np.sign(current_prediction - previous_prediction)
        actual_direction = np.sign(current_actual - previous_actual)

        # Check if the signs match
        if prediction_direction == actual_direction:
            correct_predictions += 1

        total_predictions += 1

    direction_accuracy = (correct_predictions / total_predictions) * 100

    return direction_accuracy

def visualize_results(stock_data, predictions):
    df = pd.DataFrame(predictions, columns =['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
    df.head()
    plt.figure(figsize=(15, 8))
    plt.title('Stock Prices History')
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Prices ($)')
    plt.savefig('guess.png')
    return

# Extract the actual values from the test data

train_data, test_data, scalar = load_data()
parameters, _, _, _ = train(train_data)
predictions = predict(parameters, test_data)

print(predictions)

# Denormalize the predictions and test data
predictions = scalar.inverse_transform(predictions)
test_data = scalar.inverse_transform(test_data)

actual_values = test_data[:, 4]  # Assuming the last column contains the target variable (stock prices)
window_size = 10

# Evaluate the predictions
# Calculate the direction accuracy
direction_acc = calculate_direction_accuracy(predictions[:, 4], actual_values, window_size)

# Print the direction accuracy
print("Direction Accuracy:", direction_acc)
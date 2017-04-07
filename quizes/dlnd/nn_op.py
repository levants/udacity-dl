"""
Created on Feb 7, 2017

@author: Levan Tsinadze
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data_path = os.path.join('datas', 'hour.csv')

current_dir = os.path.dirname(os.path.realpath(__file__))

dirs = os.path.split(current_dir)
current_dir = dirs[0]

files_path = os.path.join(current_dir, data_path)
rides = pd.read_csv(files_path)

rides.head()

rides[:24 * 10].plot(x='dteday', y='cnt')

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean) / std

# Save the last 21 days 
test_data = data[-21 * 24:]
data = data[:-21 * 24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days of the remaining data as a validation set
train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]

class NeuralNetwork(object):
  
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                       (self.hidden_nodes, self.input_nodes))
        print('weights_input_to_hidden - ', self.weights_input_to_hidden.shape)

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                       (self.output_nodes, self.hidden_nodes))
        print('weights_hidden_to_output - ', self.weights_hidden_to_output.shape)
        self.lr = learning_rate
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = sigmoid
    
    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        print('inputs - ', inputs.shape)
        print('targets - ', targets.shape)
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        print('hidden_inputs - ', hidden_inputs.shape)
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
        print('hidden_outputs - ', hidden_outputs.shape)
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        print('final_inputs - ', final_inputs.shape)
        final_outputs = final_inputs  # signals from final output layer
        print('final_outputs - ', final_outputs.shape)
        
        #### Implement the backward pass here ####
        ### Backward pass ###
        
        # TODO: Output error
        output_errors = targets - final_outputs  # Output layer error is the difference between desired target and actual output.
        print('output_errors - ', output_errors.shape)
        # TODO: Backpropagated error
        hidden_errors = np.dot(self.weights_hidden_to_output.T, output_errors)  # errors propagated to the hidden layer
        print('hidden_errors - ', hidden_errors.shape)
        hidden_grad = (hidden_outputs * (1 - hidden_outputs))  # hidden layer gradients
        print('hidden_grad - ', hidden_grad.shape)
        # TODO: Update the weights
        dot_prd = np.dot(output_errors, hidden_outputs.T)
        pts_prd = output_errors * hidden_outputs.T
        print('dot - ', dot_prd)
        print('pts - ', pts_prd)
        self.weights_hidden_to_output += self.lr * np.dot(output_errors, hidden_outputs.T)  # update hidden-to-output weights with gradient descent step
        prod_drad = np.dot(hidden_grad, inputs.T) 
        print('prod_drad - ', prod_drad.shape)
        self.weights_input_to_hidden += self.lr * np.dot((hidden_errors * hidden_grad), inputs.T)  # update input-to-hidden weights with gradient descent step
 
        
    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer
        
        # TODO: Output layer
        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer 
        
        return final_outputs

def MSE(y, Y):
    return np.mean((y - Y) ** 2)


### Set the hyperparameters here ###
epochs = 3550
learning_rate = 0.01
hidden_nodes = 30
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values,
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

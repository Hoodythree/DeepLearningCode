from layer import Layer
import numpy as np


class FCLayer(Layer):
    # init weights and bias
    
    def __init__(self, input_size, output_size):
        """ Description 

        :type input_size:
        :param input_size: neurons nums of input
    
        :type output_size:
        :param output_size: neurons nums of output
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.bias
        return self.output

    # update weights and bias
    # and compute input_error as the output_error of last layer 
    def back_propagation(self, output_error, learning_rate):
        """ Description
        :type output_error:
        :param output_error: upstream gradients

        :type learning_rate:
        :param learning_rate:

        """
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error
    
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        
        return input_error


        
        
from layer import Layer
import numpy as np


class ActivationLayer(Layer):
    # init activation and its derivative
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(input_data)
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
        input_error = output_error * self.activation_prime(self.input)
    
        return input_error


        
        
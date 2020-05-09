# network function:
# 0. choose loss
# 1. add layer
# 2. train
# 3. predict
from act_layer import ActivationLayer
from activations import tanh, tanh_prime
from fc_layer import FCLayer
from losses import mse, mse_prime


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer
    def add(self, layer):
        self.layers.append(layer)
    
    # choose loss function
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict : 重点
    # forward_prop ->... -> forward_prop -> final result
    def predict(self, input_data):
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # init output
            output = input_data[i]
            # forward propagation
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result
        
    # train/fit
    def fit(self, x_train, y_train, epochs, learning_rate):
        # samples' num
        samples = len(x_train)

        for i in range(epochs):
            # display error
            err = 0
            for j in range(samples):
            # forward popagation : cumpute ouput
                # init output
                output = x_train[j]
                # forward propagation
                for layer in self.layers:
                    output = layer.forward_propagation(output)
            # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)
            # back propagation: update weigths and bias
            # 初始化错了：初始error是loss对Y的导数
                error = self.loss_prime(output, y_train[j])
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)
            
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))


        


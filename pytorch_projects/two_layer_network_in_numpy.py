import numpy as np 

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

epochs = 400
learning_rate = 1e-6
def Relu(x):
    return np.maximum(0, x)

def Relu_prime():
    return 1.0

def mse(y_pred, y):
    return np.square(y_pred - y).sum()

# mse对向量求导
def mse_prime(y_pred, y):
    return 2.0 * (y_pred - y)

for i in range(epochs):

    # forward pass
    h = np.dot(x, w1)
    # array([[ 0.99204284, -1.30005321,  1.11786283],
    #    [-1.4967402 , -1.42870554, -0.512784  ]])
    # maximum:
    # array([[0.99204284, 0.        , 1.11786283],
    #    [0.        , 0.        , 0.        ]])
    h_relu = Relu(h)
    y_pred = np.dot(h_relu, w2)

    # compute loss
    loss = mse(y_pred, y)
    print(i+1, loss)

    # backward pass

    # intial grad
    grad_y = mse_prime(y_pred, y)
    grad_w2 = h_relu.T.dot(grad_y)
    # upstream grad
    grad_h_relu = grad_y.dot(w2.T)

    # total grad of hidden layer: total = local_grad * upstream_grad
    # upstream grad = total
    grad_h = grad_h_relu * Relu_prime()

    # 暂时无法理解
    # grad_h[h < 0] = 0
    
    # grad_w1 = input.T * upstream_grad
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2


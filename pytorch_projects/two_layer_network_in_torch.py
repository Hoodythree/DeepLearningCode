import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# set epochs and learning rate
epochs = 400
learning_rate = 1e-6

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

def mse(y_pred, y):
    return (y_pred - y).pow(2).sum().item()

# mse对向量求导
def mse_prime(y_pred, y):
    return 2.0 * (y_pred - y)

for i in range(epochs):

    # forward pass
    h = x.mm(w1)
    
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # compute loss
    loss = mse(y_pred, y)
    if i % 100 == 0:
        print(i+1, loss)

    # backward pass

    # intial grad
    grad_y = mse_prime(y_pred, y)
    grad_w2 = h_relu.t().mm(grad_y)
    # upstream grad
    grad_h_relu = grad_y.mm(w2.t())

    # total grad of hidden layer: total = local_grad * upstream_grad
    # upstream grad = total
    grad_h = grad_h_relu * 1.0

    # 暂时无法理解
    # grad_h[h < 0] = 0
    
    # grad_w1 = input.T * upstream_grad
    grad_w1 = x.t().mm(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

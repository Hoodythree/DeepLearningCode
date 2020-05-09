import numpy as np 
import pandas as pd

# fit function: y = beta0 + beta1 * x
# loss
def mse(y_pre, y_i):
    return np.sum((y_pre - y_i) ** 2) / len(y)

# initialize beta, alpha(learning rate), tolerance error
beta = np.array([1, 1])
alpha = 0.2
tol_error = 0.1

# compute gradient
def compute_grad(beta, x, y_i):
    grad = [0, 0]
    grad[0] = 2.0 * np.mean(beta[0] + beta[1] * x - y_i)
    grad[1] = 2.0 * np.mean((beta[0] + beta[1] * x - y_i) * x)
    return np.array(grad)


# update i+1th epoch beta according to ith epoch
def update_grad(beta, grad, alpha):
    beta = np.array(beta)- alpha * grad
    return beta


# read data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
max_x = max(train['id'])

x = train['id'] / max_x
y = train['questions']


# compute initial loss and loss_new
y_pre = beta[0] + beta[1] * x
loss = mse(y_pre, y)
grad = compute_grad(beta, x, y)
beta = update_grad(beta, grad, alpha)
y_pre = beta[0] + beta[1] * x
loss_new = mse(y_pre, y)

print('loss : %s ; loss_new : %s'%(loss, loss_new))
# see if condition meeted (if L2 norm < tol) 
# converge when loss_new - loss < tol_error
i = 0
while np.abs(loss_new - loss) > tol_error:
    loss = loss_new
    grad = compute_grad(beta, x, y)
    beta = update_grad(beta, grad, alpha)
    y_pre = beta[0] + beta[1] * x
    loss_new = mse(y_pre, y)
    i += 1
    print('Round %s Diff MSE %s'%(i, abs(loss_new - loss)))

print('Coef: %s \nIntercept %s'%(beta[1] / max_x, beta[0]))
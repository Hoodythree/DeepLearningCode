import numpy as np 
import pandas as pd

# fit function: y = beta0 + beta1 * x
# loss
# mes or rmse
def mse(y_pre, y_i):
    # return np.sum((y_pre - y_i) ** 2) / len(y)
    squared_err = (y_pre - y_i) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res

# initialize beta, alpha(learning rate), tolerance error
beta = np.array([1, 1])
alpha = 0.1
tol_error = 0.1

# Stochastic Gradient Decent
# random choice a sample then compute gradient
# choose x[r], y[r]
def compute_grad_SGD(beta, x, y_i):
    grad = [0, 0]
    r = np.random.randint(0, len(x))
    grad[0] = 2.0 * np.mean(beta[0] + beta[1] * x[r] - y_i[r])
    grad[1] = 2.0 * np.mean((beta[0] + beta[1] * x[r] - y_i[r]) * x[r])
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

grad = compute_grad_SGD(beta, x, y)
beta = update_grad(beta, grad, alpha)
y_pre = beta[0] + beta[1] * x
loss_new = mse(y_pre, y)

# print('loss : %s ; loss_new : %s'%(loss, loss_new))
# print('beta : {}'.format(beta))


# see if condition meeted (if L2 norm < tol) 
# converge when loss_new - loss < tol_error
i = 1
while np.abs(loss_new - loss) > tol_error:
    grad = compute_grad_SGD(beta, x, y)
    # print('grad : {}'.format(grad))
    beta = update_grad(beta, grad, alpha)
    # print('beta : {}'.format(beta))
    if i % 100 == 0:
        y_pre = beta[0] + beta[1] * x
        loss = loss_new
        loss_new = mse(y_pre, y)
        print('Round %s Diff MSE %s'%(i, abs(loss_new - loss)))
        # print('beta : {}'.format(beta))
    i += 1
    

print('Coef: %s \nIntercept %s'%(beta[1] / max_x, beta[0]))
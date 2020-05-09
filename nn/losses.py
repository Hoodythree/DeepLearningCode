import numpy as np 


def mse(y_pre, y):
    return 2.0 * np.mean((y_pre - y) ** 2)

# vector
def mse_prime(y_pre, y):
    return 1.0 * (y_pre - y) / y.size


if __name__ == '__main__':
    y = np.arange(5)
    y_pre = np.array([1, 1, 3, 4 ,5])
    print('mse : {} ; mse_prime : {} '.format(mse(y_pre, y), mse_prime(y_pre, y)))
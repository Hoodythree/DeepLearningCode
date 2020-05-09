import numpy as np

def L_vectorized(W, X, y):
    scores = W.dot(X)
    l_i = scores - scores[i] + 1
    l_i[y] = 0
    return np.sum(l_i)
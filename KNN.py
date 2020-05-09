from math import sqrt
import numpy as np

# caculate distance
def caculate_EDU_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1 - x2)))

# find nearest k neighbors
def find_k_nearest_neighbors(X_train, x, k):
    all_distance = []
    for xt in X_train:
        dis_xt = caculate_EDU_distance(xt[0], x)
        label_xt = xt[1]
        all_distance.append((dis_xt, label_xt))
    all_distance.sort(key = lambda x: x[0])
    return all_distance[:k]

# make predictions
def predicton(X_train, x, k):
    neighbors = find_k_nearest_neighbors(X_train, x, k)
    neighbors = [row[-1] for row in neighbors]
    predicton = max(set(neighbors), key=neighbors.count)
    return predicton

# generate data
def generate_test_data(nums):
    data = []
    np.random.seed(10)
    for i in range(nums):
        data.append((np.random.random(3), np.random.randint(3)))
    return data


X_train = generate_test_data(10)
x = ([0.33273564, 0.26558805, 0.92736888])
print(predicton(X_train, x, 3))


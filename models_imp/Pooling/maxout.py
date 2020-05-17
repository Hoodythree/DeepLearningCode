import torch.nn as nn
import torch

# CCMP: Cross-Channel Max Pooling 
# in Mutual-Channel Loss for Fine-Grained Image Classification
# 分组从深度方向进行pooling
class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x):
        assert x.shape[1] % self._pool_size == 0, \
            'Wrong input first dim size ({}) for Maxout({})'.format(x.shape[1], self._pool_size)
        m, i = x.view(*x.shape[:1], x.shape[1] // self._pool_size, self._pool_size, *x.shape[2:]).max(2)
        return m
# N * C * H * W
# 128 x 8 x 7 x 7
f_map = torch.Tensor(128, 8, 7, 7)

# 128 x 2 x 7 x 7
m = Maxout(4)
print(m(f_map).size())
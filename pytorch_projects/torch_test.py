# Source： https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py
import torch
import numpy as np

# 生成矩阵并指明类型
x = torch.zeros(5, 3, dtype=torch.long)

# 生成未初始化的矩阵
y = torch.empty(4, 2)

# 生成相同size的tensor
y = torch.randn_like(x, dtype=torch.float)

# 从array构造一个tensor
y = torch.tensor([2.9, 3])

# 从已有的tensor构造tensor
y = y.new_ones(6, 5)
x = 1

# 四则运算
# Addition: providing an output tensor as argument
res = torch.empty(6, 5)
torch.add(x, y, out=res)

# Addition: in-place
# in-place 操作一般以_结尾， 比如x.copy_(y), x.t_()
y.add_(x)

# numpy-like slice
# print('res [:, :2] = {}'.format(res[:, :2]))

# Resizing: If you want to resize/reshape tensor, you can use torch.view
# print('y = {}'.format(y))
z = y.view(30)

# -1指另外一维的维度 value = size / other_dimension_size
z = y.view(-1, 3)

# If you have a one element tensor, use .item() to get the value as a Python number
z = torch.randn(1)
# print('z = {}'.format(z.item()))

# Numpy数组和tensor的相互转换 : 改变一个的值两者的值同时被改变
# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), 
# and changing one will change the other.
a = torch.ones(5, 2)
# print('tensor a: {}'.format(a))

b = a.numpy()
# print('Numpy array b : {}'.format(b))

a.add_(2)
# print('tensor a: {}'.format(a))
# print('Numpy array b : {}'.format(b))

# Numpy arry -> Tensor
# Note : All the Tensors on the CPU except a CharTensor support converting to NumPy and back.
a = np.ones((2, 3))
b = torch.from_numpy(a)
np.add(a, 1, out=a)
# print('Numpy array a: {}'.format(a))
# print('tensor b : {}'.format(b))

# CUDA Tensors
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
x = torch.randn(1)
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    # move x to GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    # move z to CPU
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
import torch

# Creating the graph
x = torch.tensor(1.0, requires_grad = True)
y = torch.tensor(2.0)
z = x * y

# Displaying
for i, name in zip([x, y, z], "xyz"):
    print(f"{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n\
grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n")

# 构造反向传播图的时候不追踪历史
with torch.no_grad():
	# Check if tracking is enabled
	y = x * 2
	print(y.requires_grad) #False

x = torch.tensor([0.0, 2.0, 8.0], requires_grad = True)
y = torch.tensor([5.0 , 1.0 , 7.0], requires_grad = True)
z = x  * 2.0

# loss是矩阵时的反向传播
z.backward(y)
print(x.grad)
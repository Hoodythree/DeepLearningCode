import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 设置迭代轮数和学习率
epochs = 400
learning_rate = 1e-4

# 内置损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for i in range(epochs):
    # 从model获取输出
    y_pred = model(x)

    # 计算loss
    loss = loss_fn(y_pred, y)

    if i % 40 == 0:
        print(i, loss.item())
    
    # 反向传播前先将优化器的梯度清零
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model). This is because by default, gradients are
    # accumulated in buffers( i.e, not overwritten) whenever .backward()
    # is called. Checkout docs of torch.autograd.backward for more details.
    optimizer.zero_grad()

    # 对loss进行反向传播
    loss.backward()

    # 更新参数
    optimizer.step()
    
    

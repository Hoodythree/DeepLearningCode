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
epochs = 300
learning_rate = 1e-4

# 内置损失函数
loss_fn = torch.nn.MSELoss(reduction='sum')

# 训练
for i in range(epochs):
    # 从model获取输出
    y_pred = model(x)

    # 计算loss
    loss = loss_fn(y_pred, y)

    if i % 100 == 0:
        print(i, loss.item())
    
    # 反向传播前先将梯度清零
    model.zero_grad()

    # 对loss进行反向传播
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    

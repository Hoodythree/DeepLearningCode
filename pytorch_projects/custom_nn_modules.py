import torch

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

epochs = 400
learning_rate = 1e-4


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out ):
        # 确保父类被正确初始化
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    # 非静态方法
    def forward(self, x):
        y_h = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(y_h)
        return y_pred

    # 不需要实现backward

    
model = TwoLayerNet(D_in, H, D_out)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction='sum')

for i in range(epochs):
    # __call__ is already defined in nn.Module, will register all hooks and call your forward. 
    # That’s also the reason to call the module directly (output = model(data)) instead of model.forward(data).
    # 也就是说通过model(x)就直接调用了forward方法
    y_pred = model(x)

    # 计算loss
    loss = loss_fn(y_pred, y)
    if i % 50 == 0:
        print(i + 1, loss.item())
    
    # optimizer更新梯度
    # 先清零，再反向传播
    optimizer.zero_grad()

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()


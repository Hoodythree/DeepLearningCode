import torch

dtype = torch.float
device = torch.device('cpu')

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, dtype=dtype, device=device)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
epochs = 400

for i in range(epochs):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum() #tensor

    if i % 10 == 0:
        print(i, loss.item())

    # backward自动求导
    loss.backward()

    # 使用with_no_grad的原因：pytorch的动态计算图
    # It allows us to perform regular Python operations on tensors,
    # independent of PyTorch’s computation graph.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

    # Manually zero the gradients after updating weights
    # 因为梯度会叠加
        w1.grad.zero_()
        w2.grad.zero_()


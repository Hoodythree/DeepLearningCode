import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# Toy dataset
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 设置超参数
input_size = 1
output_size = 1
epochs = 100
learning_rate = 0.05

# 定义模型
model = nn.Linear(input_size, output_size)

# 定义criterion和optimizer 
criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
for epoch in range(epochs):
    # convert numpy to tensor
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, targets)
    
    # backward
    with torch.no_grad():
        model.zero_grad()
        loss.backward()
        optimizer.step()
        print('inputs grad : ', inputs.requires_grad)
    if epoch % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

# 测试
# Plot the graph

# tensor.detach() creates a tensor that shares storage with tensor that does not require grad. 
# It detaches the output from the computational graph. 
# So no gradient will be backpropagated along this variable.
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# Save the model checkpoint
torch.save(model.state_dict(), 'model\linear_model.ckpt')
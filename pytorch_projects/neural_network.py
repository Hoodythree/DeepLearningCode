# conv1 -> conv2 ->fc1->fc2->fc3
import torch
import torchvision
import torch.nn as nn
# what's this package
import torchvision.transforms as transforms

# device setting
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 超参数设置
epochs = 5
num_classes = 10
learning_rate = 0.001
batch_size = 100 # understand batch_size

# 下载并构建数据集
train_dataset = torchvision.datasets.MNIST(root='..\data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='..\data', train=False, transform=transforms.ToTensor())

# 测试一下是否构建成功
image, label = train_dataset[0]
print(image.size())
print(label)

# Data loader : split data into batches
# Data loader will behave like an iterator, so we can loop over it and fetch a different mini-batch every time
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) #why false?

# 网络搭建
class LuoNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # block1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # block2
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # fully-connected layer
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)# out.size : torch.Size([100, 32, 7, 7])
        
        # The input of a Pytorch Neural Network is of type 
        # [BATCH_SIZE] * [CHANNEL_NUMBER] * [HEIGHT] * [WIDTH].
        # 相当于flatten
        out = out.reshape(out.size(0), -1)# reshaped out.size : torch.Size([100, 1568])
        out = self.fc(out)
        return out

# 初始化模型（并使用GPU）
model = LuoNet(num_classes).to(device) 

# 定义优化器和loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练
# 多少个batch
total_step = len(train_loader)

for epoch in range(epochs):
    # 一个batch
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, total_step, loss.item()))

# 测试
model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        # 模型输出
        outputs = model(images)
        # 找出每一行最高的score并返回索引
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# 保存模型
torch.save(model.state_dict(), 'model.ckpt')
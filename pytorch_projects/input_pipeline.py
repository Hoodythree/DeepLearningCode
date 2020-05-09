import torch
import torchvision
# what's this package
import torchvision.transforms as transforms

# device setting

# 超参数


# 下载并构建数据集
train_dataset = torchvision.datasets.MNIST(root='..\data', train=True, transform=transforms.ToTensor(), download=True)

# 测试一下是否构建成功
image, label = train_dataset[0]
print(image.size())
print(label)

# Data loader


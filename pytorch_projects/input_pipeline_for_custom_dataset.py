import torch

# 继承torch.utils.data.Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass
    
    # 使得数据可以以索引的形式被访问，如data[i]
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        # 返回数据集的size
        return 0
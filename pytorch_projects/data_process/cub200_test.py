import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset


base_folder = '../data/CUB_200_2011/'

class CUB200(Dataset):
    def __init__(self, base_folder, transform=None, train=True):
        self.base_folder = base_folder
        self.transform = transform
        self.train = train
        images = pd.read_csv(os.path.join(base_folder, 'images.txt'), sep=' ',
                        names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(base_folder, 'image_class_labels.txt'),
                                    sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(base_folder, 'train_test_split.txt'),
                                        sep=' ', names=['img_id', 'is_training_img'])
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        if self.train:
            self.data = data[data.is_training_img == 1]
        else:
            self.data = data[data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(base_folder, 'images', sample.filepath)
        img = default_loader(path)
        # Target start from 1 in data, so shift to 0
        label = sample.target - 1
        return img, label
        
cub200 = CUB200(base_folder)
img, label = cub200[100]

print('label : {}'.format(label))
print('length of cub200 dataset : {}'.format(len(cub200)))
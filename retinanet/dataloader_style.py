import torch
from torch.utils import data
import numpy as np
class StyleDataset(data.Dataset):
    def __init__(self,train_data,train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def __getitem__(self,index):
        data, label = self.train_data[index], self.train_labels[index]

        label = torch.from_numpy(np.array(label))

        return data, label
    def __len__(self):
        return len(self.train_data)

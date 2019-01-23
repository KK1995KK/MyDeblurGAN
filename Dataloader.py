import torch
from torch.utils import data


class DataLoader(data.Dataset):
    def name(self):
        return 'DataLoader'

    def initialize(self, dataSet, batchSize, shuffle=True):
        self.dataSet = dataSet
        self.dataloader = torch.utils.data.DataLoader(
            self.dataSet,
            batch_size=batchSize,
            shuffle=shuffle
        )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataSet)
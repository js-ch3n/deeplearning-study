from torch.utils.data import Dataset
import torch


class IrisDataset(Dataset):
    def __init__(self, feature, label, transform=None):
        super().__init__()
        self.features = feature 
        self.labels = label
        self.transform = transform


    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        sample = self.features[index]
        label = self.labels[index]
        if self.transform:
            sample = self.transform(sample)

        return sample, label

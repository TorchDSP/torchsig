import numpy as np
import torch
from torch.utils.data import Dataset

from torchsig.image_datasets.transforms.impairments import normalize_image

"""
Defines a concatinated Dataset of all the given datasets joined as one dataset object.
Inputs:
    component_datasets: a list of Dataset objects which contain instances of each component, represented as (image_component: ndarray(c,height,width), class_id: int)
    balance: whether or not to balance the dataset by selecting samples from component datasets at random
"""
class ConcatDataset(Dataset):
    def __init__(self, component_datasets, balance = True, transforms = []):
        self.component_datasets = component_datasets
        self.balance = balance
        self.transforms = transforms
    def __len__(self):
        if self.balance:
            return max([len(ds) for ds in self.component_datasets])*len(self.component_datasets) #assume the length wed get if we upsampled other datasets to max the largest component dataset
        else:
            return np.sum([len(ds) for ds in self.component_datasets])
    def __getitem__(self, idx):
        x = None
        if self.balance: # ignores given index; will always be random with replacement
            ds = self.component_datasets[np.random.randint(0, len(self.component_datasets))]
            x = ds[np.random.randint(0, len(ds))]
        else:
            ds_ind = 0
            ds = self.component_datasets[ds_ind]
            ds_idx = idx
            while(ds_idx >= len(ds)):
                ds_idx -= len(ds)
                ds_ind += 1
                ds = self.component_datasets[ds_ind]
            x = ds[ds_idx]
        if self.transforms:
            if type(self.transforms) == list:
                for transform in self.transforms:
                    x = transform(x)
            else:
                x = self.transforms(x)
        return x
    def next(self):
        return self[np.random.randint(0,len(self))]



import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets

#source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class SplitMNIST(Dataset):
    """Split MNIST dataset."""

    def __init__(self):
        pass
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass

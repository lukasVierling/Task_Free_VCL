import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets
import torch.nn.functional as F

#source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class generateMNIST(Dataset):
    """Premuted MNIST dataset."""

    def __init__(self, train=True, label=0):
        """
        Input:
        - permutation: a permutation matrix to permute the pixels of the MNIST images
        - split is True if we use train False if we use test data
        """
        self.train = train
        self.data =  torchvision.datasets.MNIST(root='./data', train=self.train, download=True, transform=transforms.ToTensor())
        self.label = label
        print(f"Create a dataset with only MNIST images of label: {label}")
        #filter for the given  label
        boolean_mask = self.data.targets == label
        self.data.data, self.data.targets = self.data.data[boolean_mask], self.data.targets[boolean_mask]

    def get_label(self):
        return self.label
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x,y = self.data[idx]
        x_flat = x.view(-1)
        #preprocess with sigmoid <- already between 0 and 1

        return x_flat, y

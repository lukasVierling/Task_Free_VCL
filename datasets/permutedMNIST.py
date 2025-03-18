import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.datasets

#source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class PermutedMNIST(Dataset):
    """Premuted MNIST dataset."""

    def __init__(self, train=True, permutation=None):
        """
        Input:
        - permutation: a permutation matrix to permute the pixels of the MNIST images
        - split is True if we use train False if we use test data
        """
        self.train = train
        self.data =  torchvision.datasets.MNIST(root='./data', train=self.train, download=True, transform=transforms.ToTensor())
        if permutation is not None:
            if not(self.train) and permutation is None:
                print("Warning creating test set without predefined permutation")
            self.permutation = permutation
        else:
            self.permutation = torch.randperm(28*28) #permute all the entries in the image
        #print("Use the following permutation: ", self.permutation)

    def get_permutation(self):
        return self.permutation
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x,y = self.data[idx]
        x_flat = x.view(-1)
        x_permuted = x_flat[self.permutation].view(1, 28,28)

        return x_permuted, y

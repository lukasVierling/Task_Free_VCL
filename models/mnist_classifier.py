import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # hardcoded MNISTClassifier
        self.conv1 = nn.Conv2d(1,32,3)
        self.conv2 = nn.Conv2d(32,64,3)
        self.linear1 = nn.Linear(64*5*5, 128)
        self.linear2 = nn.Linear(128,10)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, 28, 28)
        h_1 = F.max_pool2d(F.relu(self.conv1(x)),2)
        h_2 = F.max_pool2d(F.relu(self.conv2(h_1)),2)
        h_2 = h_2.view(batch_size, -1)
        h_3 = F.relu(self.linear1(h_2))
        h_4 = self.linear2(h_3)
        return h_4

    def get_probs(self,x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

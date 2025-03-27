import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchvision import datasets, transforms
from tqdm import tqdm
import json
import os
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_tasks = 10
num_epochs = 10
batch_size = 256
lr = 0.001
hidden_dim = 100  # Hidden layer dimension

# Permuted MNIST Dataset
class PermutedMNIST(Dataset):
    def __init__(self, train=True, task_id=0, perm=None):
        self.mnist = datasets.MNIST(root='./data', train=train, download=True, transform=transforms.ToTensor())
        if perm is None:
            self.permutation = torch.randperm(784).to(device) if task_id > 0 else torch.arange(784).to(device)
        else:
            self.permutation = perm
            
    def get_perm(self):
        return self.permutation


    def __getitem__(self, index):
        img, label = self.mnist[index]
        img = img.view(-1)  # Flatten the image
        img = img.to(device)

        img = img[self.permutation]  # Apply permutation
        return img, label

    def __len__(self):
        return len(self.mnist)

# MLP Model with 2 hidden layers
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Synaptic Intelligence (SI)
class SI:
    def __init__(self, model, damping=0.1, xi=0.1):
        self.model = model
        self.damping = damping  # Damping factor for stability
        self.xi = xi  # Importance scaling factor
        self.omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()}  # Importance weights
        self.prev_params = {n: p.clone().detach() for n, p in model.named_parameters()}  # Previous parameters

    def update_omega(self, task_data, optimizer):
        dataloader = DataLoader(task_data, batch_size=batch_size, shuffle=True)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    delta = p - self.prev_params[n]
                    omega_update = p.grad ** 2 / (delta ** 2 + self.damping)
                    self.omega[n] += omega_update
        # Update prev_params after all batches
        optimizer.zero_grad()
        for n, p in self.model.named_parameters():
            self.prev_params[n] = p.clone().detach()

    def penalty(self):
        loss = 0
        for n, p in self.model.named_parameters():
            # Check the current parameter
            
            # Compute the difference
            diff = p - self.prev_params[n]
            
            # Square the difference
            squared_diff = diff ** 2
            
            # Multiply by omega
            weighted_diff = self.omega[n] * squared_diff
            
            # Sum the contribution for this parameter
            param_penalty = weighted_diff.sum()
            
            loss += param_penalty
        
        # Scale by xi
        total_penalty = self.xi * loss
        return total_penalty

# Laplace Propagation (LP)
class LP:
    def __init__(self, model, lambd=1.0):
        self.model = model
        self.lambd = lambd  # Regularization strength
        self.prev_params = None
        self.fisher = None

    def update_fisher(self, task_data, batch_size):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        dataloader = DataLoader(task_data, batch_size=batch_size, shuffle=True)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad ** 2
        for n in fisher:
            fisher[n] /= len(dataloader)  # Normalize by number of batches
        self.fisher = fisher
        self.prev_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

    def penalty(self):
        if self.prev_params is None:
            return 0
        loss = 0
        for n, p in self.model.named_parameters():
            delta = p - self.prev_params[n]
            loss += (self.fisher[n] * delta ** 2).sum()
        return self.lambd * loss

# Elastic Weight Consolidation (EWC)
class EWC:
    def __init__(self, model, lambd=1.0):
        self.model = model
        self.lambd = lambd  # Regularization strength
        self.prev_params = None
        self.fisher = None

    def update_fisher(self, task_data, batch_size):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        dataloader = DataLoader(task_data, batch_size=batch_size, shuffle=True)
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad ** 2
        for n in fisher:
            fisher[n] /= len(dataloader)  # Normalize by number of batches
        self.fisher = fisher
        self.prev_params = {n: p.clone().detach() for n, p in self.model.named_parameters()}

    def penalty(self):
        if self.prev_params is None:
            return 0
        loss = 0
        for n, p in self.model.named_parameters():
            delta = p - self.prev_params[n]
            loss += (self.fisher[n] * delta ** 2).sum()
        return self.lambd * loss

# Training function for one task
def train_task(model, task_data, optimizer, cl_method, epochs=10):
    dataloader = DataLoader(task_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
            if cl_method is not None:
                loss += cl_method.penalty()  # Add regularization term
            loss.backward()
            optimizer.step()

# Evaluation function
def evaluate(model, task_data):
    dataloader = DataLoader(task_data, batch_size=batch_size, shuffle=False)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

# Main method with saving results
def main(method='SI'):
    # Load 10 Permuted MNIST tasks

    train_tasks = [PermutedMNIST(train=True, task_id=i) for i in range(num_tasks)]
    test_tasks = [PermutedMNIST(train=False, task_id=i, perm=train_tasks[i].get_perm()) for i in range(num_tasks)]

    # Initialize model (single head)
    model = MLP(hidden_dim=hidden_dim).to(device)

    # Continual learning method configuration
    if method == 'SI':
        cl_method = SI(model, damping=0.1, xi=0.001)
        config = {
            'method': 'SI',
            'num_tasks': num_tasks,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'damping': 0.1,
            'xi': 0.001
        }
    elif method == 'LP':
        cl_method = LP(model, lambd=0.5)
        config = {
            'method': 'LP',
            'num_tasks': num_tasks,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'lambd': 0.5
        }
    elif method == 'EWC':
        cl_method = EWC(model, lambd=100)
        config = {
            'method': 'EWC',
            'num_tasks': num_tasks,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'lambd': 1000
        }
    else:
        raise ValueError("Invalid method. Choose 'SI', 'LP', or 'EWC'.")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    all_accuracies = []

    # Train and evaluate on each task
    for task_id in range(num_tasks):
        print(f"\nTraining on Task {task_id}")
        train_task(model, train_tasks[task_id], optimizer, cl_method, epochs=num_epochs)

        # Update continual learning method
        if method == 'SI':
            cl_method.update_omega(train_tasks[task_id], optimizer)
        elif method in ['LP', 'EWC']:
            cl_method.update_fisher(train_tasks[task_id], batch_size)

        # Evaluate on all tasks seen so far
        print(f"Evaluating after Task {task_id}:")
        accuracies = []
        for test_task in test_tasks[:task_id + 1]:
            accuracies.append(evaluate(model, test_task))
        all_accuracies.append(accuracies)
        for t, acc in enumerate(accuracies):
            print(f"Task {t} accuracy: {acc:.4f}")

    # Save results with config
    results = {
        'config': config,
        'accuracies': all_accuracies  # List of lists: accuracies[task][previous_task]
    }
    os.makedirs('results', exist_ok=True)
    filename = f'results/{method}_permuted_mnist.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filename}")

if __name__ == '__main__':
    # Run for all three methods
    for method in ['EWC']:
        print(f"\nRunning {method}")
        main(method)
# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import yaml

import sys
import os

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#my imports
from VI import coreset_vcl
from models.model import DiscriminativeModel
from datasets.permutedMNIST import PermutedMNIST

def parse_config(config_path):
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print("Error while loading config occrued")
    return config


def main(config_path):
    config = parse_config(config_path)
    print("Finished reading config")
    #dataset
    num_tasks = config["num_tasks"]
    print(f"Dataset parameters: \n  num_tasks:{num_tasks}")

    #training
    epochs = config["epochs"]
    lr = config["lr"]
    hidden_dim = config["hidden_dim"]
    batch_size = config["batch_size"]
    print(f"Training parameters:\n  epochs:{epochs}\n   hidden_dim:{hidden_dim}\n   lr:{lr}\n   batch_size:{batch_size}")

    #coreset
    coreset_size = config["coreset_size"]
    coreset_heuristic = config["coreset_heuristic"]
    print(f"VI parameters for coreset:\n   coreset_size:{coreset_size} \n   coreset_heuristic:{coreset_heuristic}")
    train_tasks = []
    test_tasks = []

    #load datasets
    for _ in range(num_tasks):
        train_tasks.append(PermutedMNIST(train=True)) 
        perm = train_tasks[-1].permutation
        test_tasks.append(PermutedMNIST(train=False, permutation=perm))
    print(f"Finished generating {num_tasks} tasks")

    #constants for MNIST
    input_dim = 28*28
    output_dim = 10 

    model = DiscriminativeModel(input_dim, output_dim, hidden_dim)

    print(f"Generated model with input_dim: {input_dim} and output_dim: {output_dim} \n Model: {model}")

    print("Start Training...")
    coreset_vcl(model, train_tasks, test_tasks, coreset_size = coreset_size, coreset_heuristic=coreset_heuristic, batch_size=batch_size, epochs=epochs, lr=lr)

    print("Finished Training!")

    print("Start Testing the final model...")

    model.eval()
    results = {}
    head =0
    with torch.no_grad():
        for test_dataset in test_tasks:
            test_loader = DataLoader(test_dataset, batch_size=128)
            correct = 0
            model.activate_head(head)
            for x,y in test_loader:
                probs = model(x)
                prediction = torch.argmax(probs, dim=-1)
                correct += (prediction==y).sum()
            acc = correct.item() / len(test_dataset)
            results[1+head] = acc
            print(f"Achieved accuracy of {acc*100}% on task {head+1} using head {head}")
            head += 1

if __name__=="__main__":
    main("configs/permutedMNIST.yaml")
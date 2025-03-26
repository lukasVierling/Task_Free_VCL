# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import yaml
import argparse
import json

import sys
import os
import matplotlib.pyplot as plt

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#my imports
from extension.algorithm import vcl
from extension.model import DiscriminativeModel as VI_model
from datasets.permutedMNIST import PermutedMNIST

def parse_config(config_path):
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print("Error while loading config occrued")
    return config

import os
import matplotlib.pyplot as plt

def plot_results(results, save_folder="extension_logs/plots"):
    #print(results)
    os.makedirs(save_folder, exist_ok=True)
    
    # Convert tensors to CPU scalars/lists:
    train_losses = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in results["train_losses"]]
    mutual_infos = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in results["mutual_info"]]
    means = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in results["means"]]
    stds = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in results["stds"]]
    
    # Handle heads_chosen: if it's a list, assume the final element is the dictionary
    if isinstance(results["heads_chosen"], list):
        if len(results["heads_chosen"]) > 0 and isinstance(results["heads_chosen"][-1], dict):
            heads_chosen_stat = results["heads_chosen"][-1]
        else:
            heads_chosen_stat = {}
    else:
        heads_chosen_stat = results["heads_chosen"]
    
    # Convert keys to int if necessary
    heads = [int(k) if isinstance(k, torch.Tensor) else k for k in heads_chosen_stat.keys()]
    counts = list(heads_chosen_stat.values())
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Batches")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "train_loss.png"))
    plt.close()
    
    # Plot mutual information
    plt.figure(figsize=(10, 6))
    plt.plot(mutual_infos, label="Mutual Information")
    plt.xlabel("Batch Index")
    plt.ylabel("MI")
    plt.title("Mutual Information Over Batches")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "mutual_information.png"))
    plt.close()
    
    # Plot baseline mean and std
    plt.figure(figsize=(10, 6))
    plt.plot(means, label="Baseline Mean")
    plt.plot(stds, label="Baseline STD")
    plt.xlabel("Batch Index")
    plt.ylabel("Value")
    plt.title("Baseline Mean and STD Over Batches")
    plt.legend()
    plt.savefig(os.path.join(save_folder, "baseline_mean_std.png"))
    plt.close()
    
    # Plot head selection statistics as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(heads, counts)
    plt.xlabel("Head Index")
    plt.ylabel("Selection Count")
    plt.title("Head Selection Statistics")
    plt.savefig(os.path.join(save_folder, "head_selection.png"))
    plt.close()
    
    # Plot final test accuracy as a bar chart (assumes results["acc"] is a scalar)
    final_acc = results["acc"]
    if isinstance(final_acc, list):
        final_acc = np.mean(final_acc)
    elif isinstance(final_acc, torch.Tensor):
        final_acc = final_acc.cpu().item()

    if isinstance(final_acc, torch.Tensor):
        final_acc = final_acc.cpu().item()
    plt.figure(figsize=(6, 6))
    plt.bar(["Final Accuracy"], [final_acc])
    plt.ylabel("Accuracy")
    plt.title("Final Test Accuracy")
    plt.savefig(os.path.join(save_folder, "final_accuracy.png"))
    plt.close()



def main(config_path, id="0", save=True):

    print(f"Save run under id:{id} and save run is: {save}")

    #constants for MNIST
    input_dim = 28*28
    output_dim = 10 

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

    #algorithm
    algorithm_name = config["algorithm_name"]
    if algorithm_name == "VI":
        model_class = VI_model
        CL_algorithm = vcl
        #coreset
        c = config["c"]
        baseline_window = config["baseline_window"]
        current_window = config["current_window"]
        num_samples = config["num_samples"]
        mode = config["mode"]
        single_head = config["single_head"]
        print(f"VI parameters for coreset:\n   baseline_window:{baseline_window} \n   current_window:{current_window}\n    mode: {mode} \n    single head:{single_head}\n    num samples: {num_samples}")
        alg_args = {"baseline_window_size":baseline_window, "current_window_size": current_window, "c": c, "num_samples": num_samples}
        model_args = {"mode": mode, "single_head": single_head}

    train_tasks = []
    test_tasks = []

    #load datasets
    for _ in range(num_tasks):
        train_tasks.append(PermutedMNIST(train=True)) 
        perm = train_tasks[-1].permutation
        test_tasks.append(PermutedMNIST(train=False, permutation=perm))
    print(f"Finished generating {num_tasks} tasks")

    #check for device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training no device:", device)

    model = model_class(input_dim, output_dim, hidden_dim, **model_args)

    print(f"Generated model with input_dim: {input_dim} and output_dim: {output_dim} \n Model: {model}")

    print("Start Training...")
    metrics = CL_algorithm(model, train_tasks, test_tasks, batch_size=batch_size, epochs=epochs, lr=lr, device=device, **alg_args)

    print("Finished Training!")

    #print(f"Average Accuracies: \n {accs}")
    accs = metrics["acc"]
    print("Final accs:", accs)
    result_dict = {
        "config": config,
        "accuracies": accs
    }
    os.makedirs('extension_logs', exist_ok=True)
    #save the accs
    if save:
        with open(f'extension_logs/{algorithm_name}_{id}.json', 'w') as f:
            json.dump(result_dict, f, indent=4)


    plot_results(metrics)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='?', default=None, help='config path')
    parser.add_argument('--id', nargs='?', default=None, help='run id')
    parser.add_argument('--no_save', action='store_true', help='Set this flag to save the model')

    args = parser.parse_args()
    config_path = args.config if args.config is not None else "configs/discriminative/lp_permutedMNIST.yaml"
    id = args.id
    save = not(args.no_save)
    default_path = "configs/discriminative/lp_permutedMNIST.yaml"
    main(config_path, id, save)


# library imports
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset
import yaml
import argparse
import json

import sys
import os

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#my imports
from Discriminative_Algorithms.VI_algorithm import coreset_vcl
from Discriminative_Algorithms.EWC_algorithm import ewc 
from Discriminative_Algorithms.SI_algorithm import si 
from Discriminative_Algorithms.LP_algorithm import lp
from models.VI_model import DiscriminativeModel as VI_model
from models.EWC_model import DiscriminativeModel as EWC_model
from models.SI_model import DiscriminativeModel as SI_model
from models.LP_model import DiscriminativeModel as LP_model
from datasets.permutedMNIST import PermutedMNIST

def parse_config(config_path):
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print("Error while loading config occrued")
    return config


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
        CL_algorithm = coreset_vcl
        #coreset
        coreset_size = config["coreset_size"]
        coreset_heuristic = config["coreset_heuristic"]
        print(f"VI parameters for coreset:\n   coreset_size:{coreset_size} \n   coreset_heuristic:{coreset_heuristic}")
        alg_args = {"coreset_size":coreset_size, "coreset_heuristic": coreset_heuristic}
    elif algorithm_name == "EWC":
        model_class = EWC_model
        CL_algorithm = ewc
        lambdas = config["lambdas"]
        print(f"EWC parameters:\n   lambdas:{lambdas}")
        alg_args = {"lambdas": lambdas}
    elif algorithm_name =="SI":
        model_class = SI_model
        CL_algorithm = si
        c = config["c"]
        damping_param = config["damping_param"]
        print(f"SI parameters:\n    c:{c}\n    damping_param:{damping_param}")
        alg_args = { "damping_param":damping_param, "c": c}
    elif algorithm_name =="LP":
        model_class = LP_model
        CL_algorithm = lp
        lambd = config["lambd"]
        print(f"LP parameters:\n    lambda:{lambd}")
        alg_args = {"lambd":lambd}
        

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

    model = model_class(input_dim, output_dim, hidden_dim)

    print(f"Generated model with input_dim: {input_dim} and output_dim: {output_dim} \n Model: {model}")

    print("Start Training...")
    accs = CL_algorithm(model, train_tasks, test_tasks, batch_size=batch_size, epochs=epochs, lr=lr, device=device, **alg_args)

    print("Finished Training!")

    print(f"Average Accuracies: \n {accs}")

    acc_list = [ acc.item() for acc in accs]
    result_dict = {
        "config": config,
        "accuracies": acc_list
    }
    os.makedirs('100_epochs', exist_ok=True)
    #save the accs
    if save:
        with open(f'100_epochs/{algorithm_name}_{id}.json', 'w') as f:
            json.dump(result_dict, f, indent=4)


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
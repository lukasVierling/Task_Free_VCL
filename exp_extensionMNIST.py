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
from collections import defaultdict
import pandas as pd
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#my imports
from extension.algorithm import vcl
from extension.expert_gate_algorithm import vcl as gated_vcl
from extension.regression_model import BayesianNN as regression_model
from extension.gating_model import DiscriminativeModel as gated_model
from extension.softmax_model import DiscriminativeModel as bernoulli_model
from datasets.permutedMNIST import PermutedMNIST
import random

def parse_config(config_path):
    with open(config_path) as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print("Error while loading config occrued")
    return config

import os
import matplotlib.pyplot as plt

def main(config_path, id="0", save=True):
    #make stuff deterministic

    config = parse_config(config_path)
    print("Finished reading config")
    seed = config["seed"]
    id = config["id"]
    save_folder = config["save_folder"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = f'{seed}'

    print(f"Save run under id:{id} and save run is: {save}")

    #constants for MNIST
    input_dim = 28*28
    output_dim = 10 

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
        CL_algorithm = vcl
        #coreset
        c = config["c"]
        baseline_window = config["baseline_window"]
        current_window = config["current_window"]
        num_samples = config["num_samples"]
        mode = config["mode"]
        single_head = config["single_head"]
        var = config["var"]
        claculation_mode = config["calculation_mode"]
        routing_mode = config["routing_mode"]
        automatic_detection = config["automatic_detection"]
        print(f"VI parameters for coreset:\n   baseline_window:{baseline_window} \n   current_window:{current_window}\n    mode: {mode} \n    single head:{single_head}\n    num samples: {num_samples} \n    var:{var}\n    calc mode: {claculation_mode}\n    routing_mode : {routing_mode}")
        alg_args = {"baseline_window_size":baseline_window, "current_window_size": current_window, "c": c, "num_samples": num_samples, "calculation_mode": claculation_mode, "routing_mode": routing_mode, "var":var, "automatic_detection":automatic_detection}
        model_args = {"mode": mode, "single_head": single_head}
        if mode == "regression":
            model_class = regression_model
        elif mode == "bernoulli":
            model_class = bernoulli_model
        #if mode == "regression":
         #   output_dim = 1
    if algorithm_name=="gated_VI":
        CL_algorithm = gated_vcl
        model_class = gated_model
        #coreset
        routing_mode = config["routing_mode"]
        autoencoder_hidden_dim = config["autoencoder_hidden_dim"]
        print(f"VI parameters for coreset:\n  routing_mode : {routing_mode}\n    autoencoder dim: {autoencoder_hidden_dim}")
        alg_args = {"routing_mode": routing_mode}
        model_args = {"autoencoder_hidden_dim": autoencoder_hidden_dim}
        
        #if mode == "regression":
         #   output_dim = 1


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
    model = model_class(input_dim=input_dim,hidden_dim=hidden_dim, output_dim=output_dim, **model_args)

    print(f"Generated model with input_dim: {input_dim} and output_dim: {output_dim} \n Model: {model}")

    print("Start Training...")
    metrics = CL_algorithm(model, train_tasks, test_tasks, batch_size=batch_size, epochs=epochs, lr=lr, device=device, **alg_args)

    print("Finished Training!")

    #print(f"Average Accuracies: \n {accs}")
    accs = metrics["acc"]
    print("Final accs:", accs)
    result_dict = {
        "config": config,
        "accuracies": accs,
        "metrics": metrics
    }
    os.makedirs(f'{save_folder}', exist_ok=True)
    #save the accs
    if save:
        with open(f'{save_folder}/{algorithm_name}_{id}.json', 'w') as f:
            json.dump(result_dict, f, indent=4, default=lambda o: o.tolist() if isinstance(o, torch.Tensor) else o)


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


#!/bin/bash
# run_experiments.sh
export CUDA_VISIBLE_DEVICES=1

# Define arrays of hyperparameters for multiple runs
lambd_values=(100 100 100 100 100 100 100 100 100 100)
seed_values=(100 101 102 103 104 105 106 107 108 109)
ids=(0 1 2 3 4 5 6 7 8 9)

# Base config file path
base_config="configs/discriminative/ewc_permutedMNIST.yaml"

# Directory to save the modified config files
modified_config_dir="configs/discriminative/modified_configs"
mkdir -p "${modified_config_dir}"

# Loop over hyperparameter sets (IDs will start from 0)
for i in "${!lambd_values[@]}"; do
    lambd=${lambd_values[$i]}
    seed=${seed_values[$i]}
    id=${ids[$i]}

    # Create a list of 10 lambd values (e.g., [0.1, 0.1, ..., 0.1])
    lambdas="["
    for j in {1..10}; do
        lambdas+="${lambd}, "
    done
    # Remove the trailing comma and space, then close the bracket
    lambdas="${lambdas%, }]"
    
    # Create a modified config file for this run
    modified_config="${modified_config_dir}/ewc_permutedMNIST_run_${i}.yaml"
    cp "${base_config}" "${modified_config}"
    
    # Update parameters in the YAML config.
    # It is assumed that the YAML file has lines beginning with "lambd:", "lambdas:", "seed:" and "id:".
    sed -i "s/^lambdas:.*/lambdas: ${lambdas}/" "${modified_config}"
    sed -i "s/^seed:.*/seed: ${seed}/" "${modified_config}"
    sed -i "s/^id:.*/id: ${id}/" "${modified_config}"
    
    # Launch the experiment in the background.
    # This command runs the experiment with the modified config file.
    python exp_permutedMNIST.py --config "${modified_config}" &
    
    echo "Launched run ${id} with lambd=${lambd}, lambdas=${lambdas}, and seed=${seed}"
done

# Wait for all background processes to finish
wait
echo "All experiments finished."

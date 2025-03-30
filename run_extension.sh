#!/bin/bash
# run_experiments.sh

export CUDA_VISIBLE_DEVICES=4

# Define arrays of hyperparameters for multiple runs
hidden_dims=(256 256 256 256 256)
seed_values=(100 101 102 103 104)
ids=(20 21 22 23 24)

# Base config file path
base_config="configs/discriminative/extension_permutedMNIST.yaml"

# Directory to save the modified config files
modified_config_dir="configs/discriminative/modified_configs"
mkdir -p "${modified_config_dir}"

# Loop over hyperparameter sets (IDs will start from 0)
for i in "${!hidden_dims[@]}"; do
    hidden_dim=${hidden_dims[$i]}
    seed=${seed_values[$i]}
    id=${ids[$i]}

    # Create a modified config file for this run
    modified_config="${modified_config_dir}/extension_permutedMNIST_run_${i}.yaml"
    cp "${base_config}" "${modified_config}"
    
    # Update parameters in the YAML config.
    sed -i "s/^hidden_dim:.*/hidden_dim: ${hidden_dim}/" "${modified_config}"
    sed -i "s/^seed:.*/seed: ${seed}/" "${modified_config}"
    sed -i "s/^id:.*/id: ${id}/" "${modified_config}"
    
    # Launch the experiment in the background.
    python exp_extensionMNIST.py --config "${modified_config}" &
    
    echo "Launched run ${id} with hidden_dim=${hidden_dim}, and seed=${seed}"
done

# Wait for all background processes to finish
wait
echo "All experiments finished."

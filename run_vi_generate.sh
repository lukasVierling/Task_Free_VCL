#!/bin/bash
# run_experiments_VI.sh
export CUDA_VISIBLE_DEVICES=4

# Define arrays of hyperparameters for multiple runs
coreset_sizes=(0 0 0 0 0 0 0 0 0 0)
seed_values=(100 101 102 103 104 105 106 107 108 109)
ids=(0 1 2 3 4 5 6 7 8 9)

# Base config file path for VI experiments
base_config="configs/generative/vi_generateMNIST.yaml"

# Directory to save the modified config files
modified_config_dir="configs/generative/modified_configs"
mkdir -p "${modified_config_dir}"

# Loop over hyperparameter sets
for i in "${!coreset_sizes[@]}"; do
    coreset_size=${coreset_sizes[$i]}
    seed=${seed_values[$i]}
    id=${ids[$i]}

    # Create a modified config file for this run
    modified_config="${modified_config_dir}/vi_generateMNIST_run_${i}.yaml"
    cp "${base_config}" "${modified_config}"
    
    # Update the coreset_size, seed, and id parameters in the config
    sed -i "s/^coreset_size:.*/coreset_size: ${coreset_size}/" "${modified_config}"
    sed -i "s/^seed:.*/seed: ${seed}/" "${modified_config}"
    sed -i "s/^id:.*/id: ${id}/" "${modified_config}"
    
    # Launch the experiment in the background
    python exp_generateMNIST.py --config "${modified_config}" &
    
    echo "Launched run ${id} with coreset_size=${coreset_size} and seed=${seed}"
done

# Wait for all background processes to finish
wait
echo "All experiments finished."
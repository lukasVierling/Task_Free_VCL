#!/bin/bash
# run_experiments_SI.sh
export CUDA_VISIBLE_DEVICES=2

# Define arrays of hyperparameters for multiple runs
c_values=(0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5)
seed_values=(100 101 102 103 104 105 106 107 108 109)
ids=(0 1 2 3 4 5 6 7 8 9)

# Base config file path for SI experiments
base_config="configs/discriminative/si_permutedMNIST.yaml"

# Directory to save the modified config files
modified_config_dir="configs/discriminative/modified_configs"
mkdir -p "${modified_config_dir}"

# Loop over hyperparameter sets (IDs will start from 0)
for i in "${!c_values[@]}"; do
    c=${c_values[$i]}
    seed=${seed_values[$i]}
    id=${ids[$i]}

    # Create a modified config file for this run
    modified_config="${modified_config_dir}/si_permutedMNIST_run_${i}.yaml"
    cp "${base_config}" "${modified_config}"
    
    # Update the c and seed parameters in the config.
    # This assumes your YAML file has lines beginning with "c:", "seed:" and "id:".
    sed -i "s/^c:.*/c: ${c}/" "${modified_config}"
    sed -i "s/^seed:.*/seed: ${seed}/" "${modified_config}"
    sed -i "s/^id:.*/id: ${id}/" "${modified_config}"
    
    # Launch the experiment in the background.
    python exp_permutedMNIST.py --config "${modified_config}" &
    
    echo "Launched run ${id} with c=${c} and seed=${seed}"
done

# Wait for all background processes to finish
wait
echo "All experiments finished."

#!/bin/bash

set -e

config_file="$1"
dataconfig_path="$2"

if [[ -z "$config_file" || -z "$dataconfig_path" ]]; then
    echo "Usage: $0 <yaml_config_file> <dataconfig_file_or_directory>"
    echo "  yaml_config_file: Path to the YAML training configuration file"
    echo "  dataconfig_file_or_directory: Path to a single .pkl file or directory containing .pkl files"
    echo ""
    echo "Examples:"
    echo "  $0 runs/train/smalltrain.yaml registry/coreplacesets/labelmixing"
    echo "  $0 runs/train/smalltrain.yaml path/to/single/dataconfig.pkl"
    exit 1
fi

# Check if config file exists and is a YAML file
if [[ ! -f "$config_file" ]]; then
    echo "Error: Config file '$config_file' does not exist."
    exit 1
fi

if [[ "$config_file" != *.yaml ]]; then
    echo "Error: Config file must be a .yaml file."
    exit 1
fi

# Function to run training for a single config and dataconfig
run_training() {
    local config_file="$1"
    local dataconfig_file="$2"
    
    echo "Training with config: $config_file"
    echo "Using dataconfig: $dataconfig_file"
    python train.py "$config_file" --dataconfig "$dataconfig_file"
}

# Check if dataconfig_path is a file or directory
if [[ -f "$dataconfig_path" ]]; then
    # Single .pkl file
    if [[ "$dataconfig_path" != *.pkl ]]; then
        echo "Error: Dataconfig file must be a .pkl file."
        exit 1
    fi
    run_training "$config_file" "$dataconfig_path"
elif [[ -d "$dataconfig_path" ]]; then
    # Directory - find all .pkl files recursively
    echo "Searching for .pkl files in: $dataconfig_path"
    pkl_files=$(find "$dataconfig_path" -type f -name "*.pkl")
    
    if [[ -z "$pkl_files" ]]; then
        echo "Error: No .pkl files found in directory '$dataconfig_path'"
        exit 1
    fi
    
    echo "Found $(echo "$pkl_files" | wc -l) .pkl file(s):"
    echo "$pkl_files" | sed 's/^/  /'
    echo ""
    
    # Run training for each .pkl file
    echo "$pkl_files" | while read -r pkl_file; do
        echo "=========================================="
        run_training "$config_file" "$pkl_file"
        echo ""
    done
else
    echo "Error: Dataconfig path '$dataconfig_path' does not exist."
    exit 1
fi

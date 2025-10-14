#!/bin/bash

set -e

input_path="$1"

if [[ -z "$input_path" ]]; then
    echo "Usage: $0 <yaml_file_or_directory>"
    exit 1
fi

if [[ -f "$input_path" && "$input_path" == *.yaml ]]; then
    python curate.py "$input_path"
elif [[ -d "$input_path" ]]; then
    find "$input_path" -type f -name "*.yaml" | while read -r yaml_file; do
        python curate.py "$yaml_file"
    done
else
    echo "Error: Input must be a .yaml file or a directory."
    exit 1
fi


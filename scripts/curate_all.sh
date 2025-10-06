#!/bin/bash

set -e

for config in runs/curate/*.yaml; do
    config_base=$(basename "$config" .yaml)
    pkl_file="registry/coreplacesets/${config_base}.pkl"
    if [ -f "$pkl_file" ]; then
        echo "Skipping $config because $pkl_file exists."
        continue
    fi
    echo "Running curate.py with $config"
    python curate.py "$config" || {
        echo "Error running curate.py with $config"
        exit 1
    }
done

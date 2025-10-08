set -e

for config in registry/coreplacesets/*/dataconfig.pkl; do
    echo "Running train.py with $config"
    python train.py runs/train/fasttrain.yaml --dataconfig "$config" || {
        echo "Error running train.py with $config"
        exit 1
    }
done

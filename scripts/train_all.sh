
for config in registry/coreplacesets/*labelmix*/dataconfig.pkl; do
    echo "Running train.py with $config"
    python train.py runs/train/midtrain.yaml --dataconfig "$config" || {
        echo "Error running train.py with $config"
        exit 1
    }
done



python curate.py runs/curate/hard-class-percentile-[0-90].yaml

python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/hard-class-percentile-[0-90].pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/poshard-class-percentile-[0-90].pkl

python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/hard-class-percentile-[50-75].pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/hard-class-percentile-[25-50].pkl
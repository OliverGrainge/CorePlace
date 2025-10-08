

python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[0-50]-cls[5000]-inst[6]/dataconfig.pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[0-50]-cls[7500]-inst[7]/dataconfig.pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[0-50]-cls[10000]-inst[8]/dataconfig.pkl

python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[25-75]-cls[5000]-inst[6]/dataconfig.pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[25-75]-cls[7500]-inst[7]/dataconfig.pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[25-75]-cls[10000]-inst[8]/dataconfig.pkl

python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[50-100]-cls[5000]-inst[6]/dataconfig.pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[50-100]-cls[7500]-inst[7]/dataconfig.pkl
python train.py runs/train/fasttrain.yaml --dataconfig registry/coreplacesets/sum-hardness-percentile[50-100]-cls[10000]-inst[8]/dataconfig.pkl
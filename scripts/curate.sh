

#python cureate.py runs/curate/baseline-cls[1250]-inst[4].yaml
#python curate.py runs/curate/baseline-cls[2500]-inst[4].yaml
#python curate.py runs/curate/baseline-cls[3750]-inst[6].yaml
#python curate.py runs/curate/baseline-cls[5000]-inst[8].yaml 

#python curate.py runs/curate/hard-class-percentile-[0-25].yaml
#python curate.py runs/curate/hard-class-percentile-[75-100].yaml
#python curate.py runs/curate/hard-class-percentile-[25-50].yaml
#python curate.py runs/curate/hard-class-percentile-[50-75].yaml

#python curate.py runs/curate/easy-class-cls[2500]-inst[4].yaml
#python curate.py runs/curate/easy-class-cls[1250]-inst[4].yaml
#python curate.py runs/curate/easy-class-cls[3750]-inst[6].yaml
python curate.py runs/curate/easy-class-cls[5000]-inst[8].yaml

python curate.py runs/curate/hard-instance-percentile-[0-25].yaml
python curate.py runs/curate/hard-instance-percentile-[75-100].yaml
python curate.py runs/curate/hard-instance-percentile-[25-50].yaml
python curate.py runs/curate/hard-instance-percentile-[50-75].yaml


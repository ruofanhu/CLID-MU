corruption_type=${1:-unif}
corruption_prob=${2:-0.4}
meta_lr=${3:-0.4}
lr=${4:-0.4}
tau=${5:-0.2}
dataset=${6:cifar10}
seed=${7:-2021}
meta_bsz=${8:-500}
meta_goal=${9:-ce}

for corruption_type in human_worst
do
for corruption_prob in 0
do
for meta_lr in 0.01    #3e-04 #0.01
do
for lr in 0.02      #0.02
do
for tau in 0.5
do
for dataset in cifar10
do
for seed in 5 6
do
for meta_bsz in 100
do
for meta_goal in clid
do
for warmup in 30
do
sbatch run_cifar_plain.sh ${corruption_type} ${corruption_prob} ${meta_lr} ${lr} ${tau} ${dataset} ${seed} ${meta_bsz} ${meta_goal} ${warmup}
done
done
done
done
done
done
done
done
done
done

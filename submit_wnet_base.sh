corruption_type=${1:-unif}
corruption_prob=${2:-0.4}
meta_lr=${3:-0.4}
lr=${4:-0.4}
tau=${5:-0.2}
dataset=${6:cifar10}
seed=${7:-2021}
meta_bsz=${8:-500}
meta_goal=${9:-ce}
scheduler=${10:-cos}
Tmax=${11:-5}
for corruption_type in unif
do
for corruption_prob in 0.6
# 0.4
do
for meta_lr in 1e-03    #3e-04
do
for lr in 0.05
do
for tau in 0.5
do
for dataset in cifar100
do
for seed in 15
do
for meta_bsz in 100
do
for meta_goal in ce_noisy
do
for scheduler in cos
do
for Tmax in 20
do
sbatch run_cifar_wnet.sh ${corruption_type} ${corruption_prob} ${meta_lr} ${lr} ${tau} ${dataset} ${seed} ${meta_bsz} ${meta_goal} ${scheduler} ${Tmax}
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
done
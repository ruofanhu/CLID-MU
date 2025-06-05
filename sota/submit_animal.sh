meta_lr=${1:-0.4}
lr=${2:-0.4}
tau=${3:-0.2}
seed=${4:-2021}
meta_bsz=${5:-500}
meta_goal=${6:-ce}
warmup=${7:-0}
model=${8:-vgg}
cos_lr=${9:-False}

#80 epoch 40 divide10
for meta_lr in 0.01  #0.0001   #3e-04 #0.01
do
for lr in 0.1     #0.02
do
for tau in 0.5
do
for seed in 10
do
for meta_bsz in 100
do
for meta_goal in clid
do
for warmup in 10
do
for model in vgg
do
for cos_lr in True
do
sbatch scripts/run_animal.sh ${meta_lr} ${lr} ${tau} ${seed} ${meta_bsz} ${meta_goal} ${warmup} ${model} ${cos_lr}
done
done
done
done
done
done
done
done
done

meta_lr=${1:-0.4}
lr=${2:-0.4}
tau=${3:-0.2}
seed=${4:-2021}
meta_bsz=${5:-500}
meta_goal=${6:-ce}
warmup=${7:-30}
split=${8:-red_noise_nl_0.8}
for meta_lr in 0.01    #3e-04 #0.01
do
for lr in 0.02      #0.02
do
for tau in 0.5
do
for seed in 8
do
for meta_bsz in 100
do
for meta_goal in clid
do
for warmup in 30
do
for split in red_noise_nl_0.4
do
sbatch run_redmini.sh ${meta_lr} ${lr} ${tau} ${seed} ${meta_bsz} ${meta_goal} ${warmup} ${split}
done
done
done
done
done
done
done
done


#!/bin/bash
#SBATCH --job-name=wn		# job name
#SBATCH --mem 40G		# memory size
#SBATCH -n 5 			# number of cpu
#SBATCH -N 1 			# number of node
#SBATCH --gres=gpu:1		# number of gpu
##SBATCH -C V100|A100|A100-80G|A100-mig|T4			# name of gpu that you want to use
#SBATCH -C A100|A100-80G|H100|H200			# name of gpu that you want to use
##SBATCH -w gpu-3-01		# name of node that you want to request for
#SBATCH -p short		# maximum running time, long (7 days) or short (1 day)
#SBATCH -t 24:00:00
##SBATCH -o H-output     	# STDOUT
##SBATCH -e H-error      	# STDERR



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
python main_wnet.py --corruption_type=${corruption_type} --corruption_prob=${corruption_prob} --lr=${lr} --meta_lr=${meta_lr} --tau=${tau} --dataset=${dataset} --gpuid 0 --seed=${seed} --meta_bsz=${meta_bsz} --meta_goal=${meta_goal} --scheduler=${scheduler} --Tmax=${Tmax}

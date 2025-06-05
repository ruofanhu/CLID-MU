#!/bin/bash
#SBATCH --job-name=vri_d		# job name
#SBATCH --mem 40G		# memory size
#SBATCH -n 5 			# number of cpu
#SBATCH -N 1 			# number of node
#SBATCH --gres=gpu:1		# number of gpu
##SBATCH -C V100|A100|A100-80G|A100-mig|T4			# name of gpu that you want to use
#SBATCH -C A100|A100-80G|H100|V100			# name of gpu that you want to use
##SBATCH -w gpu-3-01		# name of node that you want to request for
#SBATCH -p long		# maximum running time, long (7 days) or short (1 day)
#SBATCH -t 168:00:00
##SBATCH -o H-output     	# STDOUT
##SBATCH -e H-error      	# STDERR
##SBATCH --mail-type=BEGIN,END	# notifications for job done & fail
#SBATCH --mail-user=rhu@wpi.edu 


corruption_type=${1:-unif}
corruption_prob=${2:-0.4}
meta_lr=${3:-0.4}
lr=${4:-0.4}
tau=${5:-0.2}
dataset=${6:cifar10}
seed=${7:-2021}
meta_bsz=${8:-500}
meta_goal=${9:-ce}
warmup=${10:-10}
python Train_cifar_vri.py --noise_mode=${corruption_type} --r=${corruption_prob} --lr=${lr} --meta_lr=${meta_lr} --tau=${tau} --dataset=${dataset} --seed=${seed} --meta_bsz=${meta_bsz} --meta_goal=${meta_goal} --warmup=${warmup}
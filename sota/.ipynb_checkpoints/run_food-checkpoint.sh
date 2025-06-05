#!/bin/bash
#SBATCH --job-name=vri_cf		# job name
#SBATCH --mem 80G		# memory size
#SBATCH -n 5 			# number of cpu
#SBATCH -N 1 			# number of node
#SBATCH --gres=gpu:1		# number of gpu
##SBATCH -C V100|A100|A100-80G|A100-mig|T4			# name of gpu that you want to use
#SBATCH -C A100|A100-80G|H200			# name of gpu that you want to use
##SBATCH -w gpu-3-01		# name of node that you want to request for
#SBATCH -p short		# maximum running time, long (7 days) or short (1 day)
#SBATCH -t 24:00:00
##SBATCH -o H-output     	# STDOUT
##SBATCH -e H-error      	# STDERR
##SBATCH --mail-type=BEGIN,END	# notifications for job done & fail
#SBATCH --mail-user=rhu@wpi.edu 



meta_lr=${1:-0.4}
lr=${2:-0.4}
tau=${3:-0.2}
seed=${4:-2021}
meta_bsz=${5:-500}
meta_goal=${6:-ce}
warmup=${7:-10}
module load cuda12.1
python Train_food101n_vri.py --lr=${lr} --meta_lr=${meta_lr} --tau=${tau} --seed=${seed} --meta_bsz=${meta_bsz} --meta_goal=${meta_goal} --warmup=${warmup}
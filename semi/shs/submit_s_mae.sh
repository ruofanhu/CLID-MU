for alg in flexmatch_lossnet
# uda_lossnet fixmatch_lossnet flexmatch_lossnet
do
for label_amount in 4000
do
for noise_type in human_worst 
# human_aggre human_random
do
for noise_ratio in 0
do
for meta_lr in 0.001
do
for seed in 2
do
for meta_loss in cer
do
cfg_name=b_${alg}_cifar10_wrn_28_2_${label_amount}_lr0.03_True_bsz100_${noise_type}_${noise_ratio}_False_1000_100_${meta_loss}_uni_${meta_lr}_beta0_${seed}.yaml
sbatch ./jobs/run_s.sh ${alg} ${cfg_name}
echo ${cfg_name}
done
done
done
done
done
done
done

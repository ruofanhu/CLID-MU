for alg in flexmatch_lossnet 
# fixmatch_lossnet uda_lossnet
do
for label_amount in 4000
do
for noise_type in human_worst
# human_worst human_aggre human_random
do
for noise_ratio in 0
do
for meta_lr in 1e-05
do
for seed in 3 
do
for meta_loss in clid 
do
for e_bsz in 100
do
cfg_name=b_0.1_${alg}_cifar10_wrn_28_2_${label_amount}_lr0.03_True_bsz100_${noise_type}_${noise_ratio}_False_10000_${e_bsz}_${meta_loss}_uni_${meta_lr}_beta0_${seed}.yaml
bash ./jobs/run_s.sh ${alg} ${cfg_name}

echo ${cfg_name}
done
done
done
done
done
done
done
done

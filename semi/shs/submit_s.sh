for alg in flexmatch_lossnet 
# fixmatch_lossnet
do
for label_amount in 4000
do
for noise_type in human_worst
# human_worst human_aggre human_random
do
for noise_ratio in 0
do
for meta_lr in 1e-05
# 5e-05 1e-05
do
for seed in 3 
# 3 8 11 15 
do
for meta_loss in feat_expno1N
do
for e_bsz in 100
do
# b_fullysupervised_lossnet_cifar10_wrn_28_10_50000_lr0.1_True_bsz100_sym_0.4_False_10000_1000_feat_expno1N_uni_0.0001_beta0_8.yaml
# cfg_name=m_uda_cifar10_resnet18_${label_amount}_lr0.0005_False_bsz128_${noise_type}_${noise_ratio}_False_8960_1280_${meta_goal}_${nor}_${threshold}_10.yaml
cfg_name=b_0.1_${alg}_cifar10_wrn_28_2_${label_amount}_lr0.03_True_bsz100_${noise_type}_${noise_ratio}_False_10000_${e_bsz}_${meta_loss}_uni_${meta_lr}_beta0_${seed}.yaml
bash ./jobs/run_s.sh ${alg} ${cfg_name}
# sbatch ./jobs/ru.sh ${alg} ${cfg_name}

echo ${cfg_name}
done
done
done
done
done
done
done
done

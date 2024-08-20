for alg in fullysupervised_lossnet
do

for label_amount in 50000
do
for noise_type in sym
# human_aggre human_random
do
for noise_ratio in 0.6
do
for meta_lr in 0.001
# 0.001
#1e-05
# 5e-05
do
for seed in 15
# 3 8 11 15
do
for meta_loss in None
# feat_expno1N hfeat_expno1N
do
for lr in 0.01
do
# b_fullysupervised_lossnet_cifar10_wrn_28_10_50000_lr0.03_True_bsz100_sym_0.6_False_1000_100_None_uni_0.001_beta0_2.
# b_fullysupervised_lossnet_cifar10_wrn_28_10_50000_lr0.1_True_bsz100_sym_0.4_False_10000_1000_feat_expno1N_uni_0.0001_beta0_8.yaml
# cfg_name=m_uda_cifar10_resnet18_${label_amount}_lr0.0005_False_bsz128_${noise_type}_${noise_ratio}_False_8960_1280_${meta_goal}_${nor}_${threshold}_10.yaml
cfg_name=b_${alg}_cifar10_wrn_28_10_${label_amount}_lr${lr}_True_bsz100_${noise_type}_${noise_ratio}_False_1000_100_${meta_loss}_uni_${meta_lr}_beta0_${seed}.yaml
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
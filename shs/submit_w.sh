for alg in fullysupervised_lossnet
do 
for dataset in agnews
do
for lr in 2e-05
do
for meta_lr in 1e-05
do
for meta_loss in cer
# cer cen feat_expno1N
do
for esize in 12000
do
for seed in 2
do
for th in 0.1
# 0.2 0.3 0.4 0.5 0.6
do

cfg_name=${alg}_${th}_${dataset}_lr${lr}_bsz32_${esize}_32_${meta_loss}_${meta_lr}_${seed}.yaml
sbatch jobs/run_w.sh ${alg} ${cfg_name}
echo ${cfg_name}

done
done
done
done
done
done
done
done
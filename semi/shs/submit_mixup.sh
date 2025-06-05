for alg in fixmatch uda flexmatch
do
for dataset in cifar100
do
for label_amount in 10000
do
for noise_type in sym
do
for noise_ratio in 0.5 0.2
do
for seed in 12
do

cfg_name=mixup_0.5_${alg}_${dataset}_${label_amount}_${noise_type}_${noise_ratio}_${seed}.yaml
sbatch run_usb.sh ${alg} ${cfg_name}
echo ${cfg_name}

done
done
done
done
done
done
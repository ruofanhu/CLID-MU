# CLIDMU
© [2024] [Anonymous]. All rights reserved.
This code is provided for review purposes only. No license is granted for use, distribution, or modification of this code at this time.
## Create config files
First generate the configuration for the experiments. the 'create_config_xxx.ipynb' files are all for configuration generation but for different datsets and settigns.
## Run experiment


```bash

# replace the alg and cfg_name
alg=udametalearning
cfg_name=udametalearnin.yaml
python train.py --c config/classic_cv_meta/${alg}/${cfg_name}
```
You can reproduce our experiment by running the scripts in the **shs** folder.
```bash
# semi-supervised learing experiments
bash shs/submit_s.sh
```
```bash
# learing with noisy labels experiments
bash shs/submit_.sh
```
## Analysis
The analysis code and results can be found in the **analysis** folder.
## Data
Need to download data and put into the **data** folder.
CIFAR-10/100,CIFAR-10N,Clothing1M

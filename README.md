
# Cross-Layer Information Divergence Based Meta Update Strategy

[![KDD 2025](https://img.shields.io/badge/KDD-2025-blue)](https://dl.acm.org/doi/10.1145/3711896.3736880) [![Paper](https://img.shields.io/badge/Paper-ACM-orange)](https://dl.acm.org/doi/10.1145/3711896.3736880) [![arXiv](https://img.shields.io/badge/arXiv-2507.11807-b31b1b.svg)](https://arxiv.org/abs/2507.11807) [![DOI](https://zenodo.org/badge/843203955.svg)](https://doi.org/10.5281/zenodo.15595960) 

Official implementation of **CLID-MU**, accepted at **KDD 2025**. 


## TL;DR 

Meta-learning methods for **learning with noisy labels (LNL)** often rely on a small clean meta-dataset to guide training. However, obtaining trusted clean labels can be expensive or infeasible in real-world settings. 

**CLID-MU removes this requirement.** 

Our key observation is that **clean samples better preserve structural consistency across the last hidden layer and the final prediction layer, whereas noisy samples tend to disrupt this consistency**. CLID-MU measures this **Cross-Layer Information Divergence** and uses it as a label-free signal to guide meta updates. **No clean meta-data → Cross-layer structural consistency → Robust meta-learning under label noise**


## Environment Setup

This project uses a Conda environment to manage dependencies. To set up the environment, follow the steps below:


### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ruofanhu/CLID-MU.git
   cd CLID-MU
2. **Install the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate clidmu

## Dataset

The datasets used in our experiments can be accessed via the following Google Drive link:

🔗 [Download Dataset](https://drive.google.com/drive/folders/1MwFyCJE0SofZ-hK2SZC_bxyt-kP3q1K4?usp=drive_link)

Put all datasets into the 'data' folder.

## Run
1. **VRI**:
   
   ```bash
   #CLID-MU
   bash submit_vri_clid.sh
   ```
   
   ```bash
   #Baselines
   bash submit_vri_base.sh
   ```   
2. **WNet**
   ```bash
   #CLID-MU
   bash submit_wnet_clid.sh
   ```
   ```bash
   #Baselines
   bash submit_wnet_base.sh
   ```
## SSL and SOTA
The code for integrating CLID-MU with semi-supervised learning (SSL) and state-of-the-art (SOTA) methods can be found in the *semi* and *sota* folders, respectively.


## Acknowledgments

This project builds upon and incorporates code from the following repositories:

- [VRI](https://github.com/haolsun/VRI) – for VRI implementation.
- [meta-weight-net](https://github.com/xjtushujun/meta-weight-net) – for WNet implementation.
- [Semi-supervised-learning](https://github.com/microsoft/Semi-supervised-learning) - for SSL implementation

We thank the original authors for making their code publicly available.

 

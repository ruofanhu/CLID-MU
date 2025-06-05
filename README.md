
# Cross-Layer Information Divergence Based Meta Update Strategy

[![DOI](https://zenodo.org/badge/843203955.svg)](https://doi.org/10.5281/zenodo.15595960)


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

 

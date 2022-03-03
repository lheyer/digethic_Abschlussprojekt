# Digethic-Abschlussprojekt: Incorporating Prior Scientific Knowledge Into Deep Learning

>This repository is based on the papers "Physics Guided RNNs for Modeling Dynamical Systems: A Case Study in Simulating Lake Temperature Profiles" (Jia, Xiaowei, et al., 2019, https://epubs.siam.org/doi/abs/10.1137/1.9781611975673.63 ) and "Process-Guided Deep Learning Predictions of Lake Water Temperature" (Read et al., 2019, https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019WR024922 ) and its data and code (https://doi.org/10.5066/P9AQPIVD, https://zenodo.org/record/3497495#.YgGLqe8xnJE).




Prediction of Lake Temperatures with physics-guided Deep Learning (PGDL)


## Visual Studio Code

This repository is optimized for [Visual Studio Code](https://code.visualstudio.com/) 

## Installation of virtual environment

### Linux and Mac Users

- run the setup script `./setup.sh` or `sh setup.sh`

### Windows Users

- run the setup script `.\setup.ps1`

## Development

- activate python environment: `source .venv/bin/activate`
- run python script: `python <srcfilename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`

## Get data with git lfs

### Linux users (Debian & Ubuntu)

**Installation**

`sudo apt install git-lfs`

Then run in the repository folder:

`git lfs install`

**Pull data**

`git lfs pull`

## Run scripts

**Pretraining**

`python src/pretrain.py`

**Training**

`python scr/train.py -d 'similar' -pretrain True`

- -d flag for choosing train dataset from ['similar','season','year']
- -pretrain flag for choosing to use pretrained model [True,False](False is default)





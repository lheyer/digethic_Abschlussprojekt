# Digethic-Abschlussprojekt: Incorporating Prior Scientific Knowledge IntoDeep Learning

This repository is based on the Physics Guided RNNs for Modeling Dynamical Systems: A Case Study in Simulating Lake Temperature Profiles




Prediction of Lake Temperatures with physics-guided Deep Learing (PGDL)


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





#!/usr/bin/env bash

# Install essentials
sudo apt-get update

sudo apt-get install -y emacs htop tree git dstat

# Install linuxbrew
sudo apt-get install -y python-dev python-pip nginx

# Get Miniconda and make it the main Python interpreter
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh

echo "PATH=\$PATH:\$HOME/miniconda/bin" >> .bash_profile

#may need somehting to reload the .bash_profile so that conda is immediately available
CONDA_ENV_NAME=trxsfr-learning-web-app
$HOME/miniconda/bin/conda create -vy -n $CONDA_ENV_NAME python=3.5 flask
source activate $CONDA_ENV_NAME
pip install -r ~/trxsfr-learning-web-app/requirements.txt

#!/usr/bin/env bash

CONDA_ENV_NAME=trxsfr-learn-web

#need to make this verbose
conda create -y -n $CONDA_ENV_NAME python=3.5

#need to test the below lines
source activate $CONDA_ENV_NAME
pip install -r requirements.txt

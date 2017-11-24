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

# Set-Up Stack Driver

curl -O https://repo.stackdriver.com/stack-install.sh
sudo bash stack-install.sh --write-gcm

curl -sSO https://dl.google.com/cloudagents/install-logging-agent.sh
sudo bash install-logging-agent.sh
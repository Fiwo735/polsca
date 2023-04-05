#!/usr/bin/env bash

# Environment set in Dockerfile
source ~/.bashrc

# Environemnt for Xilinx tools
source /scratch/shared/Xilinx/Vitis_HLS/2020.2/settings64.sh

# Needed for PyPhism
export PYTHONPATH=$(pwd)
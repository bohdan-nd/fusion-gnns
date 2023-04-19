#!/bin/bash
#SBATCH --job-name=kfold_split
#SBATCH --output=logs/kfold_split.log
#SBATCH --qos=normal
#SBATCH --time=00:10:00
#SBATCH --partition=cpu
#SBATCH --mem=16G

export PYTHONPATH=${PWD}
python -u cross_validation/kfold_split.py --dataset_name oneil --fold_number 10 --validation_ratio 0.1
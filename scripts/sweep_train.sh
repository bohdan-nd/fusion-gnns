#!/bin/bash
#SBATCH --job-name=graph_%a
#SBATCH --output=logs/graph/graph_%a.log
#SBATCH --qos=normal
#SBATCH --time=15:00:00
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-11

export PYTHONPATH=${PWD}
wandb agent uoft-research-2023/drug_synergy/1nvmhrtk
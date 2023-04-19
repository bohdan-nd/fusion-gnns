#!/bin/bash
#SBATCH --job-name=drugbank_ddi
#SBATCH --output=logs/drugbank_ddi.log
#SBATCH --qos=normal
#SBATCH --time=15:00:00
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

export PYTHONPATH=${PWD}
python -u drugbank_gnn.py --fold_number 0 --synergy_score loewe --num_layers 5 --inject_layer 3 --keep_original_context true --initialization Bert
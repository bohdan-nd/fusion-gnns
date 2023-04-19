from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import argparse
from constants import *


def split_k_fold(data_folder, synergy_score, fold_number, validation_ratio):
    folds = {}

    drugcomb = pd.read_feather(f"{data_folder}/{synergy_score}.feather")
    drugcomb_indexes = drugcomb.index.values
    drugcomb_targets = drugcomb[TARGET_COLUMN_NAME].values

    skfold = StratifiedKFold(n_splits=fold_number, shuffle=True, random_state=RANDOM_SEED)

    for index, (train_indexes, test_indexes) in enumerate(skfold.split(drugcomb_indexes, drugcomb_targets)):
        train_set, val_set = train_test_split(train_indexes, test_size=validation_ratio, random_state=RANDOM_SEED)
        folds[f"fold_{index}"] = {"train": train_set.tolist(),
                                  "val": val_set.tolist(),
                                  "test": test_indexes.tolist()}

    with open(f"{data_folder}/{FOLDS_FOLDER_NAME}/{synergy_score}.json", "w") as file:
        json.dump(folds, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split K Folds')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--fold_number', type=int, default=5)
    parser.add_argument('--validation_ratio', type=float, default=0.1)

    args = parser.parse_args()

    dataset_name, fold_number, validation_ratio = args.dataset_name, args.fold_number, args.validation_ratio

    if dataset_name == "drugcomb":
        data_folder = DRUGCOMB_DATA_FOLDER
    elif dataset_name == "oneil":
        data_folder = ONEIL_DATA_FOLDER

    for synergy_score in SYNERGY_SCORES:
        split_k_fold(data_folder, synergy_score, fold_number, validation_ratio)

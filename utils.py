import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from constants import *


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def calculate_roc_auc(targets, preds):
    return roc_auc_score(targets, preds)


def calculate_auprc(targets, preds):
    precision_scores, recall_scores, __ = precision_recall_curve(targets, preds)

    return auc(recall_scores, precision_scores)


def dict_to_device(dict_, device):
    return {key: value.to(device) for key, value in dict_.items()}


def log_train_params(model):
    all_params = sum(p.numel() for p in model.parameters())
    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable params:{grad_params}/{all_params} ~ {100*(grad_params/all_params):.2f}%")


def get_double_df(df):
    col_dict = {
        'Drug1_ID': 'Drug2_ID',
        'Drug2_ID': 'Drug1_ID',
        'Drug1': 'Drug2',
        'Drug2': 'Drug1'
    }

    orig_order = df.columns
    df_rev = df.rename(columns=col_dict)
    df_rev = df_rev[orig_order]
    combo = pd.concat([df, df_rev], axis=0, ignore_index=True)
    combo = combo.drop_duplicates().reset_index(drop=True)
    return combo


def split_fold(dataset, fold: dict[str, list[int]]):
    train_indices, val_indices, test_indices = fold["train"], fold["val"], fold["test"]
    X_train = dataset.iloc[train_indices]
    X_val = dataset.iloc[val_indices]
    X_test = dataset.iloc[test_indices]

    return X_train, X_val, X_test

def create_train_sampler(X_train, weighted_sampler=False):
    if weighted_sampler:
        train_class_count = X_train[TARGET_COLUMN_NAME].value_counts().values
        train_weight = 1. / train_class_count
        train_samples_weight = np.array([train_weight[int(t)] for t in X_train[TARGET_COLUMN_NAME].values])
        train_sampler = WeightedRandomSampler(train_samples_weight, len(train_samples_weight), replacement=True)
    else:
        train_sampler = None

    return train_sampler

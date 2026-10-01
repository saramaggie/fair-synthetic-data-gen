import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SPLIT_DIR = "data/split"
SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split(df, target="is_recid", test_size=0.2, seed=SEED):
    df = df.reset_index(drop=True)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    assert set(train_df.index).isdisjoint(test_df.index), "train/test overlap"

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train_df.to_csv(f"{SPLIT_DIR}/real_train.csv", index=True)
    test_df.to_csv(f"{SPLIT_DIR}/real_test.csv", index=True)

    print(f"train: {len(train_df)} rows | test: {len(test_df)} rows")
    print("reoffending rate by sex (train):")
    print(train_df.groupby("sex")[target].mean().round(3))
    return train_df, test_df


def load_split():
    train_df = pd.read_csv(f"{SPLIT_DIR}/real_train.csv", index_col=0)
    test_df = pd.read_csv(f"{SPLIT_DIR}/real_test.csv", index_col=0)
    return train_df, test_df


def check_same_test_set(old_test_index, test_df):
    return set(old_test_index) == set(test_df.index)


def distance_to_closest_record(synth_X, train_X, test_X):
    from sklearn.neighbors import NearestNeighbors

    d_train = (
        NearestNeighbors(n_neighbors=1).fit(train_X).kneighbors(synth_X)[0].ravel()
    )
    d_test = NearestNeighbors(n_neighbors=1).fit(test_X).kneighbors(synth_X)[0].ravel()
    return {
        "median_dcr_train": float(np.median(d_train)),
        "median_dcr_test": float(np.median(d_test)),
    }

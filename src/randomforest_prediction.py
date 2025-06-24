import warnings
from typing import Tuple
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime
import csv


"""
****************************************************************
 Random Forest Prediction
 Note: Run standalone after randomforest_model_training.py
****************************************************************
"""


# *** Split groupwise for training/validation - Identical to XGBoost model ***
def groupwise_time_split(df: pd.DataFrame,
                         val_days: int = 28,
                         min_train_days: int = 90) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train, val = [], []
    for name, g in df.groupby(["item_id", "store_id"], sort=False, observed=True):
        g = g.sort_values("d")
        if len(g) >= min_train_days + val_days:
            train.append(g.iloc[:-val_days])
            val.append(g.iloc[-val_days:])
        else:
            warnings.warn(f"Skipping {name} – only {len(g)} rows.")
    if not train:
        raise ValueError("No groups have sufficient data for training.")
    return pd.concat(train, ignore_index=True), pd.concat(val, ignore_index=True)


# *** Implement baseline predictions ***
def compute_baseline_predictions(train_df: pd.DataFrame,
                                 val_df: pd.DataFrame,
                                 window: int = 28) -> np.ndarray:
    preds = []
    for key, v in val_df.groupby(["item_id", "store_id"], observed=True):
        g = train_df.loc[(train_df["item_id"] == key[0]) &
                         (train_df["store_id"] == key[1])].sort_values("d")
        baseline = g["sales"].mean() if len(g) < window else (
            g["sales"].rolling(window, min_periods=window).mean().iloc[-1])
        preds.extend([baseline] * len(v))
    return np.array(preds)


# TODO: load model to make our predictions
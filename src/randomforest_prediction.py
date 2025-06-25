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


# *** Split the dataset for training/validation ***
# from each product/location group we must have at least 90 training days and 28 validation days
def groupwise_time_split(df: pd.DataFrame, validation_day_count: int = 28, min_training_days: int = 90):
    training, validation = [], []
    for group_name, group in df.groupby(["item_id", "store_id"], sort=False, observed=True):
        group = group.sort_values("d")
        if len(group) >= min_training_days + validation_day_count:
            training.append(group.iloc[:-validation_day_count])
            validation.append(group.iloc[-validation_day_count:])
        else:
            print(f"Warning: skipping {group_name} – only {len(group)} rows.")
    if not training:
        raise ValueError("No groups have sufficient data for training.")
    return pd.concat(training, ignore_index=True), pd.concat(validation, ignore_index=True)


# *** Compute the naive baseline predictions ***
# from the training data, we take the last 28 days sales (rolling mean) and calculate the mean sales to use for calculating naive baseline MAE & RMSE
def compute_baseline_predictions(training_df: pd.DataFrame, validation_df: pd.DataFrame, window: int = 28):
    baseline_predictions = []
    for key, value in validation_df.groupby(["item_id", "store_id"], observed=True):
        temp_grouping = training_df.loc[(training_df["item_id"] == key[0]) & (training_df["store_id"] == key[1])].sort_values("d")
        baseline = temp_grouping["sales"].mean() if len(temp_grouping) < window else (
            temp_grouping["sales"].rolling(window, min_periods=window).mean().iloc[-1])
        baseline_predictions.extend([baseline] * len(value))
    return np.array(baseline_predictions)


# TODO: load model to make our predictions
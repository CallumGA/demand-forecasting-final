from __future__ import annotations
import os
from typing import Tuple
import h2o
import numpy as np
import pandas as pd
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
****************************************************************
  Random Forest Model Training/Eval
  Note: Run standalone before randomforest_prediction.py
****************************************************************
"""


# *** Split the dataset for training/validation ***
# from each product/location group we must have at least 90 training days and 28 validation days
def groupwise_time_split(
    df: pd.DataFrame,
    validation_day_count: int = 28,
    min_training_days: int = 90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    return (
        pd.concat(training, ignore_index=True),
        pd.concat(validation, ignore_index=True),
    )


# *** Compute the naive baseline predictions ***
# from the training data, we take the last 28 days sales (rolling mean) and calculate the mean sales to use for calculating naive baseline MAE & RMSE
def compute_baseline_predictions(
    training_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    window: int = 28,
) -> np.ndarray:
    baseline_predictions = []
    for key, value in validation_df.groupby(["item_id", "store_id"], observed=True):
        temp_grouping = training_df.loc[
            (training_df["item_id"] == key[0]) & (training_df["store_id"] == key[1])
        ].sort_values("d")
        baseline = (
            temp_grouping["sales"].mean()
            if len(temp_grouping) < window
            else temp_grouping["sales"].rolling(window, min_periods=window).mean().iloc[-1]
        )
        baseline_predictions.extend([baseline] * len(value))
    return np.array(baseline_predictions)


# *** Training and validation ***
def train_and_eval_rf(csv_file: str, target_col: str = "sales") -> None:
    print("Loading CSV …")
    training_data_df = pd.read_csv(csv_file)
    training_data_df["item_id"] = training_data_df["item_id"].astype("category")
    training_data_df["store_id"] = training_data_df["store_id"].astype("category")

    print("Group-wise time split …")
    train_df, validation_df = groupwise_time_split(training_data_df, 28, 90)

    # compute baseline predictions + metrics
    baseline_predictions = compute_baseline_predictions(train_df, validation_df, 28)
    baseline_mae = mean_absolute_error(validation_df[target_col], baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(validation_df[target_col], baseline_predictions))

    # 16gb RAM on my macbook air m3
    h2o.init(max_mem_size="16G", nthreads=-1)
    h2o_train = h2o.H2OFrame(train_df)
    h2o_validation = h2o.H2OFrame(validation_df)
    for col in ["item_id", "store_id"]:
        h2o_train[col] = h2o_train[col].asfactor()
        h2o_validation[col] = h2o_validation[col].asfactor()

    features = [c for c in h2o_train.columns if c != target_col]

    # train our DRF model
    print("Training standard DRF …")
    model = H2ORandomForestEstimator(
        ntrees=300,
        max_depth=10,
        min_rows=10,
        sample_rate=0.9,
        seed=42,
    )
    model.train(x=features, y=target_col, training_frame=h2o_train, validation_frame=h2o_validation)

    print("Predicting on validation …")
    predictions = model.predict(h2o_validation)["predict"].as_data_frame().values.ravel()

    # evaluation metrics
    y_validation = validation_df[target_col].values
    mae = mean_absolute_error(y_validation, predictions)
    rmse = np.sqrt(mean_squared_error(y_validation, predictions))

    print("\n────────── RESULTS ──────────")
    print(f"Baseline – MAE: {baseline_mae:8.4f}   RMSE: {baseline_rmse:8.4f}")
    print(f"DRF      – MAE: {mae:8.4f}   RMSE: {rmse:8.4f}")
    print("First few rows (truth | predictions):")
    for t, p in zip(y_validation[:10], predictions[:10]):
        print(f"{t:8.2f}  {p:8.2f}")
    print("──────────────────────────────")

    model_path = h2o.save_model(
        model=model,
        path="/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/models",
        force=True,
    )
    print(f"Model saved to: {model_path}")

    h2o.shutdown(prompt=False)


# *** Main entry point ***
if __name__ == "__main__":
    csv_path = os.path.expanduser("~/h2o_data/training_input_data.csv")
    train_and_eval_rf(csv_path)

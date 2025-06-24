import os
import warnings
from typing import Tuple
import numpy as np
import pandas as pd
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
****************************************************************
  Random Forest point-forecast (mean) – Group-wise split
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


# *** Training and validation ***
def train_and_eval_rf(csv_file: str,
                      target_col: str = "sales") -> None:
    print("➡️  Loading CSV …")
    pdf = pd.read_csv(csv_file)
    pdf["item_id"]  = pdf["item_id"].astype("category")
    pdf["store_id"] = pdf["store_id"].astype("category")

    print("➡️  Group-wise time split …")
    train_df, val_df = groupwise_time_split(pdf, 28, 90)

    # compute baselines
    base_pred = compute_baseline_predictions(train_df, val_df, 28)
    base_mae  = mean_absolute_error(val_df[target_col], base_pred)
    base_rmse = np.sqrt(mean_squared_error(val_df[target_col], base_pred))

    h2o.init(max_mem_size="16G", nthreads=-1)
    htrain = h2o.H2OFrame(train_df)
    hval   = h2o.H2OFrame(val_df)
    for col in ["item_id", "store_id"]:
        htrain[col] = htrain[col].asfactor()
        hval[col]   = hval[col].asfactor()

    features = [c for c in htrain.columns if c != target_col]

    # train our DRF model
    print("➡️  Training standard DRF …")
    rf = H2ORandomForestEstimator(
        ntrees      = 300,
        max_depth   = 10,
        min_rows    = 10,
        sample_rate = 0.9,
        seed        = 42
    )
    rf.train(x=features, y=target_col,
             training_frame=htrain, validation_frame=hval)

    print("➡️  Predicting on validation …")
    rf_pred = rf.predict(hval)["predict"].as_data_frame().values.ravel()

    # evaluation metrics
    y_val   = val_df[target_col].values
    rf_mae  = mean_absolute_error(y_val, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))

    print("\n────────── RESULTS ──────────")
    print(f"Baseline – MAE: {base_mae:8.4f}   RMSE: {base_rmse:8.4f}")
    print(f"DRF      – MAE: {rf_mae:8.4f}   RMSE: {rf_rmse:8.4f}")
    print("First few rows (truth | rf_pred):")
    for t, p in zip(y_val[:10], rf_pred[:10]):
        print(f"{t:8.2f}  {p:8.2f}")
    print("──────────────────────────────")

    h2o.shutdown(prompt=False)


# *** Main entry point ***
if __name__ == "__main__":
    csv_path = os.path.expanduser("~/h2o_data/training_input_data.csv")
    train_and_eval_rf(csv_path)
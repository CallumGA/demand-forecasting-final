import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

"""
****************************************************************
 Point-Forecast XGBoost trainer (with groupwise interval logic)
****************************************************************
"""


# static hyperparameters
POINT_FORECAST_CONFIG = {
    "max_depth": 7,
    "learning_rate": 0.1,
    "n_estimators": 300,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_lambda": 1.0,
    "reg_alpha": 0.1,
}


# *** Split the dataset for training/validation ***
# from each product/location group we must have at least 90 training days and 28 validation days
def groupwise_time_split(df: pd.DataFrame, val_days: int = 28, min_train_days: int = 90):
    train, val = [], []
    for name, g in df.groupby(["item_id", "store_id"], sort=False, observed=True):
        g = g.sort_values("d")
        if len(g) >= min_train_days + val_days:
            train.append(g.iloc[:-val_days])
            val.append(g.iloc[-val_days:])
        else:
            print(f"Warning: skipping {name} – only {len(g)} rows.")
    if not train:
        raise ValueError("No groups have sufficient data for training.")
    return pd.concat(train, ignore_index=True), pd.concat(val, ignore_index=True)


# *** Compute the naive baseline predictions ***
# from the training data, we take the last 28 days sales (rolling mean) and calculate the mean sales to use for calculating naive baseline MAE & RMSE
def compute_baseline_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame, window: int = 28):
    preds = []
    for key, v in val_df.groupby(["item_id", "store_id"], observed=True):
        g = train_df.loc[(train_df["item_id"] == key[0]) & (train_df["store_id"] == key[1])].sort_values("d")
        baseline = g["sales"].mean() if len(g) < window else (
            g["sales"].rolling(window, min_periods=window).mean().iloc[-1])
        preds.extend([baseline] * len(v))
    return np.array(preds)


# *** Add prediction bands for each product/location group ***
#  We use the "error" from the training model for each product/location pair and + or - from the new prediction to get the interval band for new predictions
#  Compute the ε:
#       abs_res = (yi - ŷi), where yi = actual sales from training data (per item/location), ŷi = predicted sales from training data (per item/location)
#       ε = np.quantile(abs_res, 1 - alpha / 2)
#  Derive the lower and upper bounds from ε:
#       Lower = max(0, ŷ_new − ε), Upper = (ŷ_new + ε), where ŷ_new = new predicted sales,  ε = epsilon
def add_groupwise_prediction_intervals(model, X_val, X_train, y_train, val_df, train_df,
                                       confidence_level=0.95, calib_frac=0.10):
    point = model.predict(X_val)
    lower = np.zeros_like(point)
    upper = np.zeros_like(point)
    interval_width = np.zeros_like(point)
    epsilon = {}

    alpha = 1 - confidence_level
    val_df = val_df.copy()
    val_df["pred"] = point
    val_df = val_df.reset_index(drop=True)

    for key, group in val_df.groupby(["item_id", "store_id"], observed=True):
        idx_train = ((train_df["item_id"] == key[0]) & (train_df["store_id"] == key[1]))
        X_tg = X_train[idx_train]
        y_tg = y_train[idx_train]

        if len(y_tg) < 10:
            eps = 0.0
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_tg), max(1, int(calib_frac * len(X_tg))), replace=False)
            abs_res = np.abs(y_tg.iloc[idx] - model.predict(X_tg.iloc[idx]))
            eps = np.quantile(abs_res, 1 - alpha / 2)

        group_idx = val_df[(val_df["item_id"] == key[0]) & (val_df["store_id"] == key[1])].index
        lower[group_idx] = np.maximum(0, val_df.loc[group_idx, "pred"] - eps)
        upper[group_idx] = val_df.loc[group_idx, "pred"] + eps
        interval_width[group_idx] = upper[group_idx] - lower[group_idx]
        epsilon[str(key)] = eps

    return {
        "point_forecast": point,
        "lower_bound": lower,
        "upper_bound": upper,
        "interval_width": interval_width,
        "epsilon": epsilon,
        "confidence_level": confidence_level,
    }


# *** Evaluate our interval bands to see how well they performed for evaluation ***
def evaluate_prediction_intervals(y_true, intv):
    within = (y_true >= intv["lower_bound"]) & (y_true <= intv["upper_bound"])
    return {
        "coverage": float(within.mean()),
        "average_width": float(intv["interval_width"].mean()),
        "expected_coverage": intv["confidence_level"],
    }


# *** Train the model ***
# 1. Split features and target (sales we want to predict) for both training & validation
# 2. Define the model ie; squared error, tree method, and pass in the hyperparameters
# 3. Fit the model with training data and validation data
# 4. Make our predictions
# 5. Create the interval bands and calculate evaluation metrics
def train_point_forecast_model(model_name: str = "xgb_point_forecast"):
    data_path = os.getenv("CLEANED_SALES_DATA", "sales_cleaned.csv")
    df = pd.read_csv(data_path)
    df["item_id"] = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")
    FEATURES = [
        "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
        "day_of_week", "month", "item_id", "store_id",
    ]

    train_df, val_df = groupwise_time_split(df, 28, 90)

    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_val, y_val = val_df[FEATURES], val_df["sales"]

    model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=30,
        **POINT_FORECAST_CONFIG,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    intervals = add_groupwise_prediction_intervals(model, X_val, X_train, y_train, val_df, train_df, 0.95)

    metrics = {
        "mae": mean_absolute_error(y_val, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "interval": evaluate_prediction_intervals(y_val, intervals),
    }

    if os.getenv("SAVED_MODELS"):
        os.makedirs(os.getenv("SAVED_MODELS"), exist_ok=True)
        joblib.dump(model, os.path.join(os.getenv("SAVED_MODELS"), f"{model_name}.joblib"))

    return model, metrics


# *** Entry point ***
def train_point_forecast():
    return train_point_forecast_model("xgb_point_forecast")


if __name__ == "__main__":
    train_point_forecast()
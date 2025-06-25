from __future__ import annotations
import os
from typing import Dict, Tuple, Any
import joblib
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
****************************************************************
 Point-Forecast w/ Groupwise Interval Model Trainer
 Note: Run standalone before xgboost_prediciton.py
****************************************************************
"""


# static hyperparameters
POINT_FORECAST_CONFIG: Dict[str, Any] = {
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
def groupwise_time_split(
    df: pd.DataFrame,
    validation_day_count: int = 28,
    min_training_days: int = 90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return group-wise train/validation splits respecting a minimum history."""
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
    """Return per-row baseline forecasts for the validation set."""
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


# *** Add prediction bands for each product/location group ***
#  We use the "error" from the training model for each product/location pair and + or - from the new prediction to get the interval band for new predictions
#  Compute the ε:
#       abs_res = (yi - ŷi), where yi = actual sales from training data (per item/location), ŷi = predicted sales from training data (per item/location)
#       ε = np.quantile(abs_res, 1 - alpha / 2)
#  Derive the lower and upper bounds from ε:
#       Lower = max(0, ŷ_new − ε), Upper = (ŷ_new + ε), where ŷ_new = new predicted sales,  ε = epsilon
def add_groupwise_prediction_intervals(
    model: xgboost.XGBRegressor,
    X_validation: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    validation_df: pd.DataFrame,
    train_df: pd.DataFrame,
    confidence_level: float = 0.95,
    calib_frac: float = 0.10,
) -> Dict[str, Any]:
    """Return point forecasts and calibrated prediction intervals."""
    point = model.predict(X_validation)
    lower = np.zeros_like(point)
    upper = np.zeros_like(point)
    interval_width = np.zeros_like(point)
    epsilon: Dict[str, float] = {}

    alpha = 1 - confidence_level
    validation_df = validation_df.copy()
    validation_df["pred"] = point
    validation_df = validation_df.reset_index(drop=True)

    for key, group in validation_df.groupby(["item_id", "store_id"], observed=True):
        idx_train = (train_df["item_id"] == key[0]) & (train_df["store_id"] == key[1])
        X_tg = X_train[idx_train]
        y_tg = y_train[idx_train]

        if len(y_tg) < 10:
            eps = 0.0
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_tg), max(1, int(calib_frac * len(X_tg))), replace=False)
            abs_res = np.abs(y_tg.iloc[idx] - model.predict(X_tg.iloc[idx]))
            eps = np.quantile(abs_res, 1 - alpha / 2)

        group_idx = validation_df[(validation_df["item_id"] == key[0]) & (validation_df["store_id"] == key[1])].index
        lower[group_idx] = np.maximum(0, validation_df.loc[group_idx, "pred"] - eps)
        upper[group_idx] = validation_df.loc[group_idx, "pred"] + eps
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
def evaluate_prediction_intervals(
    y_true: pd.Series,
    intv: Dict[str, Any],
) -> Dict[str, float]:
    within_bounds = (y_true >= intv["lower_bound"]) & (y_true <= intv["upper_bound"])
    return {
        "coverage": float(within_bounds.mean()),
        "average_width": float(intv["interval_width"].mean()),
        "expected_coverage": intv["confidence_level"],
    }


# ──────────────────────────── Training ─────────────────────────────── #
# *** Train the model ***
# 1. Split features and target (sales we want to predict) for both training & validation
# 2. Define the model ie; squared error, tree method, and pass in the hyperparameters
# 3. Fit the model with training data and validation data
# 4. Make our predictions
# 5. Create the interval bands and calculate evaluation metrics
def train_point_forecast_model(model_name: str = "xgb_point_forecast") -> Tuple[xgboost.XGBRegressor, Dict[str, Any]]:
    training_input_path = (
        "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/"
        "final-project/final-project-implementation/data/processed/training_input_data.csv"
    )
    training_input_df = pd.read_csv(training_input_path)
    training_input_df["item_id"] = training_input_df["item_id"].astype("category")
    training_input_df["store_id"] = training_input_df["store_id"].astype("category")
    FEATURES = [
        "sell_price",
        "is_event_day",
        "lag_7",
        "rolling_mean_7",
        "day_of_week",
        "month",
        "item_id",
        "store_id",
        "is_weekend",
    ]

    train_df, validation_df = groupwise_time_split(training_input_df, 28, 90)

    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_val, y_val = validation_df[FEATURES], validation_df["sales"]

    model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=30,
        **POINT_FORECAST_CONFIG,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_predictions = model.predict(X_val)
    interval_predictions = add_groupwise_prediction_intervals(
        model,
        X_val,
        X_train,
        y_train,
        validation_df,
        train_df,
        0.95,
    )

    metrics = {
        "mae": mean_absolute_error(y_val, y_predictions),
        "rmse": np.sqrt(mean_squared_error(y_val, y_predictions)),
        "interval": evaluate_prediction_intervals(y_val, interval_predictions),
    }

    save_path = (
        "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/"
        "MIA5130/final-project/final-project-implementation/models"
    )
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(model, os.path.join(save_path, f"{model_name}.joblib"))

    print("\nModel training complete.")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(
        f"Interval Coverage: {metrics['interval']['coverage']:.4f} (Expected: {metrics['interval']['expected_coverage']:.2f})"
    )
    print(f"📏 Avg Interval Width: {metrics['interval']['average_width']:.4f}")

    return model, metrics


# *** Entry point ***
def train_point_forecast():
    return train_point_forecast_model("xgb_point_forecast")


if __name__ == "__main__":
    train_point_forecast()

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
****************************************************************
 Point Forecasting XGBoost trainer with RMSE and MAE metrics
****************************************************************
"""

# TODO: add interval predictions
# TODO: optuna for hyperparams
# TODO: go through each function/calculation and document/explain for future
# TODO: prepare the cleaned csv for what we submit code-wise (just stripped down training code)
# TODO: Make predictions on the actual evaluation data
# TODO: graph samples and document the actual predictions
# TODO: finish report


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


def groupwise_time_split(df: pd.DataFrame, val_days: int = 28, min_train_days: int = 90):
    """
    Improved time split with minimum training period requirement
    """
    train, val = [], []
    for name, group in df.groupby(["item_id", "store_id"], sort=False, observed=True):
        group = group.sort_values("d")
        if len(group) >= min_train_days + val_days:
            train.append(group.iloc[:-val_days])
            val.append(group.iloc[-val_days:])
        else:
            print(f"Warning: Skipping group {name} - insufficient data ({len(group)} days)")

    if not train:
        raise ValueError("No groups have sufficient data for training")

    return pd.concat(train, ignore_index=True), pd.concat(val, ignore_index=True)


def compute_individual_losses(y_true, y_pred_model, y_pred_baseline):
    """
    Compute individual MAE and RMSE for each data point
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred_model = np.asarray(y_pred_model, dtype=float).ravel()
    y_pred_baseline = np.asarray(y_pred_baseline, dtype=float).ravel()

    mae_model = np.abs(y_true - y_pred_model)
    mae_baseline = np.abs(y_true - y_pred_baseline)

    se_model = (y_true - y_pred_model) ** 2
    se_baseline = (y_true - y_pred_baseline) ** 2

    return mae_model, mae_baseline, se_model, se_baseline


def compute_baseline_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame, window: int = 28):
    """
    Compute rolling mean baseline without data leakage
    """
    baseline_preds = []

    for name, val_group in val_df.groupby(["item_id", "store_id"], observed=True):
        train_group = train_df[
            (train_df["item_id"] == name[0]) &
            (train_df["store_id"] == name[1])
            ].sort_values("d")

        if len(train_group) < window:
            baseline_val = train_group["sales"].mean()
            baseline_preds.extend([baseline_val] * len(val_group))
        else:
            train_rolling = train_group["sales"].rolling(
                window=window, min_periods=window
            ).mean()

            last_baseline = train_rolling.iloc[-1]
            baseline_preds.extend([last_baseline] * len(val_group))

    return np.array(baseline_preds)


def validate_features(df: pd.DataFrame, features: list):
    """Validate that required features exist and handle missing values"""
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    for feature in features:
        if feature in ["item_id", "store_id"]:
            continue
        nan_pct = df[feature].isna().mean()
        if nan_pct > 0.1:
            print(f"Warning: {feature} has {nan_pct:.1%} missing values")

    return df.dropna(subset=features)


def train_point_forecast_model(model_name: str):
    data_path = os.getenv("CLEANED_SALES_DATA", "cleaned_sales.csv")
    df = pd.read_csv(data_path)
    df["item_id"] = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    FEATURES = [
        "sell_price",
        "is_event_day",
        "lag_7",
        "rolling_mean_7",
        "day_of_week",
        "month",
        "item_id",
        "store_id",
    ]

    df = validate_features(df, FEATURES)

    train_df, val_df = groupwise_time_split(df, val_days=28, min_train_days=90)
    print(f"Training groups: {train_df.groupby(['item_id', 'store_id'], observed=True).ngroups}")
    print(f"Validation samples: {len(val_df)}")

    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_val, y_val = val_df[FEATURES], val_df["sales"]

    cfg = POINT_FORECAST_CONFIG
    model = xgboost.XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=30,
        **cfg,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)

    baseline_preds = compute_baseline_predictions(train_df, val_df)

    mae_model = mean_absolute_error(y_val, y_pred)
    rmse_model = np.sqrt(mean_squared_error(y_val, y_pred))

    mae_baseline = mean_absolute_error(y_val, baseline_preds)
    rmse_baseline = np.sqrt(mean_squared_error(y_val, baseline_preds))

    individual_mae_model, individual_mae_baseline, individual_se_model, individual_se_baseline = \
        compute_individual_losses(y_val, y_pred, baseline_preds)

    val_df_reset = val_df.reset_index(drop=True)
    sample_size = min(100, len(val_df_reset))
    sample_indices = np.random.choice(len(val_df_reset), sample_size, replace=False)

    sample_records = []
    for idx in sample_indices:
        record = {
            "item_id": str(val_df_reset.iloc[idx]["item_id"]),
            "store_id": str(val_df_reset.iloc[idx]["store_id"]),
            "actual_sales": float(y_val.iloc[idx]),
            "model_prediction": float(y_pred[idx]),
            "baseline_prediction": float(baseline_preds[idx]),
            "model_mae": float(individual_mae_model[idx]),
            "baseline_mae": float(individual_mae_baseline[idx]),
            "model_se": float(individual_se_model[idx]),
            "baseline_se": float(individual_se_baseline[idx]),
            "mae_improvement": float(individual_mae_baseline[idx] - individual_mae_model[idx]),
            "se_improvement": float(individual_se_baseline[idx] - individual_se_model[idx])
        }
        sample_records.append(record)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}_{timestamp}.json")

    log_content = {
        "timestamp": timestamp,
        "model_name": model_name,
        "model_type": "point_forecast",
        "data_stats": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_groups": train_df.groupby(['item_id', 'store_id']).ngroups
        },
        "hyperparameters": cfg,
        "metrics": {
            "model": {
                "mae": float(mae_model),
                "rmse": float(rmse_model),
            },
            "baseline": {
                "mae": float(mae_baseline),
                "rmse": float(rmse_baseline),
            },
            "improvement": {
                "mae_reduction": float((mae_baseline - mae_model) / mae_baseline),
                "rmse_reduction": float((rmse_baseline - rmse_model) / rmse_baseline)
            }
        },
        "sample_predictions": sample_records,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_content, f, indent=2)
    print(f"[INFO] Results logged to: {log_path}")

    print(f"\n=== {model_name} (Point Forecast) ===")
    print(f"Model    - MAE: {mae_model:.3f}, RMSE: {rmse_model:.3f}")
    print(f"Baseline - MAE: {mae_baseline:.3f}, RMSE: {rmse_baseline:.3f}")
    print(f"MAE improvement: {(mae_baseline - mae_model) / mae_baseline:.1%}")
    print(f"RMSE improvement: {(rmse_baseline - rmse_model) / rmse_baseline:.1%}")

    save_dir = os.getenv("SAVED_MODELS")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, f"{model_name}.json"))

    return model


def add_prediction_intervals(model, X_val, confidence_level: float = 0.95):
    """
    Add prediction intervals to point forecasts
    """
    point_forecast = model.predict(X_val)

    # TODO: Implement interval estimation logic here
    lower_bound = None  # TODO: Implement lower bound calculation
    upper_bound = None  # TODO: Implement upper bound calculation
    interval_width = None  # TODO: Calculate interval width

    return {
        'point_forecast': point_forecast,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'interval_width': interval_width
    }


def train_point_forecast():
    model = train_point_forecast_model("xgb_point_forecast")
    return model


if __name__ == "__main__":
    train_point_forecast()
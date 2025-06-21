import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
****************************************************************
 Point-Forecast XGBoost trainer
****************************************************************
"""

# ---------------------------------------------------------------------------
# Tuned hyper-parameters for the point model
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------
def groupwise_time_split(df: pd.DataFrame,
                         val_days: int = 28,
                         min_train_days: int = 90):
    train, val = [], []
    for name, g in df.groupby(["item_id", "store_id"],
                              sort=False, observed=True):
        g = g.sort_values("d")
        if len(g) >= min_train_days + val_days:
            train.append(g.iloc[:-val_days])
            val.append(g.iloc[-val_days:])
        else:
            print(f"Warning: skipping {name} – only {len(g)} rows.")
    if not train:
        raise ValueError("No groups have sufficient data for training.")
    return pd.concat(train, ignore_index=True), pd.concat(val, ignore_index=True)

# ---------------------------------------------------------------------------
# Per-row loss decomposition
# ---------------------------------------------------------------------------
def compute_individual_losses(y_true, y_pred_model, y_pred_base):
    y_true  = np.asarray(y_true, dtype=float).ravel()
    y_pm    = np.asarray(y_pred_model, dtype=float).ravel()
    y_pb    = np.asarray(y_pred_base,  dtype=float).ravel()

    mae_m   = np.abs(y_true - y_pm)
    mae_b   = np.abs(y_true - y_pb)
    se_m    = (y_true - y_pm) ** 2
    se_b    = (y_true - y_pb) ** 2
    return mae_m, mae_b, se_m, se_b

# ---------------------------------------------------------------------------
# Rolling-mean baseline
# ---------------------------------------------------------------------------
def compute_baseline_predictions(train_df: pd.DataFrame,
                                 val_df: pd.DataFrame,
                                 window: int = 28):
    preds = []
    for key, v in val_df.groupby(["item_id", "store_id"], observed=True):
        g = train_df.loc[(train_df["item_id"] == key[0]) &
                         (train_df["store_id"] == key[1])].sort_values("d")
        baseline = g["sales"].mean() if len(g) < window else (
            g["sales"].rolling(window, min_periods=window).mean().iloc[-1])
        preds.extend([baseline] * len(v))
    return np.array(preds)

# ---------------------------------------------------------------------------
# Split-conformal prediction interval
# ---------------------------------------------------------------------------
def add_prediction_intervals(model, X_val, X_train, y_train,
                             confidence_level=0.95, calib_frac=0.10):
    point = model.predict(X_val)

    rng     = np.random.default_rng(42)
    idx     = rng.choice(len(X_train), int(calib_frac*len(X_train)), replace=False)
    abs_res = np.abs(y_train.iloc[idx] - model.predict(X_train.iloc[idx]))

    alpha = 1 - confidence_level
    eps   = np.quantile(abs_res, 1 - alpha/2)

    lower = point - eps
    upper = point + eps
    lower = np.maximum(0, lower)

    return {
        "point_forecast": point,
        "lower_bound":    lower,
        "upper_bound":    upper,
        "interval_width": upper - lower,
        "epsilon":        eps,
        "confidence_level": confidence_level,
    }

def evaluate_prediction_intervals(y_true, intv):
    within = (y_true >= intv["lower_bound"]) & (y_true <= intv["upper_bound"])
    return {
        "coverage":          float(within.mean()),
        "average_width":     float(intv["interval_width"].mean()),
        "expected_coverage": intv["confidence_level"],
    }

# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------
def train_point_forecast_model(model_name: str):
    data_path = os.getenv("CLEANED_SALES_DATA", "cleaned_sales.csv")
    df        = pd.read_csv(data_path)
    df["item_id"]  = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    FEATURES = [
        "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
        "day_of_week", "month", "item_id", "store_id",
    ]

    train_df, val_df = groupwise_time_split(df, 28, 90)
    print(f"Training groups : {train_df.groupby(['item_id','store_id'], observed=True).ngroups}")
    print(f"Validation rows : {len(val_df)}")

    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_val,   y_val   = val_df[FEATURES],   val_df["sales"]

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

    intv   = add_prediction_intervals(model, X_val, X_train, y_train, 0.95)
    intv_m = evaluate_prediction_intervals(y_val, intv)

    y_base = compute_baseline_predictions(train_df, val_df)
    mae_m, rmse_m = mean_absolute_error(y_val, y_pred), np.sqrt(mean_squared_error(y_val, y_pred))
    mae_b, rmse_b = mean_absolute_error(y_val, y_base), np.sqrt(mean_squared_error(y_val, y_base))

    rng   = np.random.default_rng(0)
    idxs  = rng.choice(len(val_df), min(100, len(val_df)), replace=False)
    samples = [{
        "item_id": str(val_df.iloc[i]["item_id"]),
        "store_id": str(val_df.iloc[i]["store_id"]),
        "actual": float(y_val.iloc[i]),
        "pred":   float(y_pred[i]),
        "base":   float(y_base[i]),
        "lb":     float(intv["lower_bound"][i]),
        "ub":     float(intv["upper_bound"][i]),
        "in_band": bool(intv["lower_bound"][i] <= y_val.iloc[i] <= intv["upper_bound"][i]),
    } for i in idxs]

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    logp = os.path.join(os.getenv("LOG_DIR", "logs"), f"{model_name}_{ts}.json")
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    with open(logp, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": ts,
            "model_name": model_name,
            "metrics": {
                "mae": mae_m,  "rmse": rmse_m,
                "baseline_mae": mae_b, "baseline_rmse": rmse_b,
                "interval": intv_m | {"epsilon": float(intv["epsilon"])},
            },
            "sample_predictions": samples,
        }, f, indent=2)
    print(f"[INFO] log written → {logp}")
    print(f"MAE {mae_m:.3f}  vs base {mae_b:.3f}  |  RMSE {rmse_m:.3f} vs {rmse_b:.3f}")
    print(f"95 % coverage {intv_m['coverage']:.3f}  (target 0.95)")

    if os.getenv("SAVED_MODELS"):
        os.makedirs(os.getenv("SAVED_MODELS"), exist_ok=True)
        model.save_model(os.path.join(os.getenv("SAVED_MODELS"), f"{model_name}.json"))

    return model

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def train_point_forecast():
    return train_point_forecast_model("xgb_point_forecast")

if __name__ == "__main__":
    train_point_forecast()
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
****************************************************************
 Quantile XGBoost trainer **with** rolling-quantile baseline
 For each quantile:
 1.	Load data → 2. group-by time split → 3. train XGBoost quantile model → 4. generate validation predictions
  → 5. build rolling-quantile baseline → 6. compute identical pinball-loss metrics for model and baseline → 7. dump everything into a timestamped JSON log (plus optional model file).
****************************************************************
"""

# ---------------------------------------------------------------------------
# hyperparameters (optuna tuned)
# ---------------------------------------------------------------------------
QUANTILE_CONFIGS = {
    0.9: {
        "max_depth": 7,
        "learning_rate": 0.11986597583433842,
        "n_estimators": 334,
        "subsample": 0.920309685792909,
        "colsample_bytree": 0.768086782088947,
        "min_child_weight": 8.249795901241658,
        "gamma": 0.4622984156928419,
        "reg_lambda": 0.5790748841440645,
        "reg_alpha": 1.0946520339010668,
    },
    0.5: {
        "max_depth": 8,
        "learning_rate": 0.11421647505386022,
        "n_estimators": 281,
        "subsample": 0.7272559090308933,
        "colsample_bytree": 0.8875348068275094,
        "min_child_weight": 8.285762749215156,
        "gamma": 1.037840631251667,
        "reg_lambda": 4.109038729104575,
        "reg_alpha": 3.184399463923405,
    },
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def groupwise_time_split(df: pd.DataFrame, val_days: int = 28):
    """Chronological split per (item, store) to train | validation."""
    train, val = [], []
    for _, g in df.groupby(["item_id", "store_id"], sort=False, observed=True):
        g = g.sort_values("d")
        if len(g) > val_days + 30:
            train.append(g.iloc[:-val_days])
            val.append(g.iloc[-val_days:])
    return pd.concat(train, ignore_index=True), pd.concat(val, ignore_index=True)


# ---------------------------------------------------------------------------
# Loss Function (Predicted & Baseline)
# ---------------------------------------------------------------------------
# Pinball loss (a.k.a. quantile loss) formula:
#     L_q(y, ŷ) = mean_i { max( q · (y_i − ŷ_i),
#                              (q − 1) · (y_i − ŷ_i) ) }
# Interpretation
#  • Under-prediction (y_i > ŷ_i)  → loss =  q · (y_i − ŷ_i)
#  • Over-prediction  (y_i ≤ ŷ_i)  → loss = (q − 1) · (y_i − ŷ_i)
#
# Asymmetric penalty forces the model to target the chosen quantile:
# higher q amplifies the cost of under-predicting, lower q amplifies the cost
# of over-predicting.
# ─────────────────────────────────────────────────────────────────────────────
def compute_pinball_loss(y_true, y_pred, q: float):
    """Vectorised pinball loss with shape-checks."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch in pinball loss: {y_true.shape} vs {y_pred.shape}"

    delta = y_true - y_pred
    return np.mean(np.maximum(q * delta, (q - 1) * delta))


def groupwise_pinball_loss(df: pd.DataFrame, q: float):
    def _loss(sub):
        sub = sub.dropna(subset=["sales", "pred"])
        if sub.empty:
            return np.nan

        # Handle accidental duplicate "pred" columns
        y_pred = sub["pred"]
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.iloc[:, 0]

        return compute_pinball_loss(sub["sales"], y_pred, q)

    return (
        df.groupby(["item_id", "store_id"], sort=False, observed=True)
        .apply(_loss, include_groups=False)
        .dropna()
        .mean()
    )


def rolling_quantile_baseline(full_df: pd.DataFrame, q: float, window: int = 28):
    return (
        full_df.groupby(["item_id", "store_id"], sort=False, observed=True)["sales"]
        .shift(1)
        .rolling(window=window, min_periods=window)
        .quantile(q)
    )


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------
def train_boosted_model(quantile_alpha: float, model_name: str):
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

    train_df, val_df = groupwise_time_split(df)
    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_val, y_val = val_df[FEATURES], val_df["sales"]

    # ---------------- Model ----------------
    cfg = QUANTILE_CONFIGS[quantile_alpha]
    model = xgboost.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        tree_method="hist",
        enable_categorical=True,
        random_state=42,
        early_stopping_rounds=30,
        **cfg,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # model predictions
    y_pred = model.predict(X_val)

    # Ensure no duplicate 'pred' columns from previous runs
    if "pred" in val_df.columns:
        val_df = val_df.drop(columns="pred")

    val_eval = val_df.copy()
    val_eval["pred"] = y_pred
    val_eval["covered"] = (val_eval["sales"] <= y_pred).astype(int)

    # -------------- Baseline --------------
    full_df = pd.concat([train_df, val_eval], ignore_index=True)
    baseline_series = rolling_quantile_baseline(full_df, quantile_alpha).loc[val_eval.index]
    val_eval["baseline_pred"] = baseline_series

    mask = ~val_eval["baseline_pred"].isna()
    val_masked = val_eval[mask].copy()

    bp = val_masked["baseline_pred"]
    if bp.ndim > 1:
        bp = bp.iloc[:, 0]

    # ---------------- Metrics -------------
    avg_cov = (
        val_eval.groupby(["item_id", "store_id"], observed=True)["covered"].mean().mean()
    )
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    pinball = compute_pinball_loss(y_val, y_pred, quantile_alpha)
    grp_pinball = groupwise_pinball_loss(val_eval, quantile_alpha)

    # Baseline (masked rows)
    baseline_mae = mean_absolute_error(val_masked["sales"], bp)
    baseline_rmse = np.sqrt(mean_squared_error(val_masked["sales"], bp))
    baseline_pinball = compute_pinball_loss(val_masked["sales"], bp, quantile_alpha)

    base_df_for_loss = val_masked.rename(columns={"baseline_pred": "pred"})
    baseline_grp_pinball = groupwise_pinball_loss(base_df_for_loss, quantile_alpha)
    baseline_cov = (
        base_df_for_loss.groupby(["item_id", "store_id"], observed=True)["covered"].mean().mean()
    )

    # ---------------- Logging -------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.getenv("LOG_DIR", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_name}_{timestamp}.txt")

    samples = (
        val_eval.groupby(["item_id", "store_id"], sort=False, observed=True)
        .apply(lambda g: g.sample(1, random_state=42))
        .reset_index(drop=True)[
            ["item_id", "store_id", "sales", "pred", "baseline_pred"]
        ]
        .to_dict(orient="records")
    )

    log_content = {
        "timestamp": timestamp,
        "model_name": model_name,
        "quantile_alpha": quantile_alpha,
        "hyperparameters": cfg,
        "metrics": {
            "model": {
                "avg_groupwise_coverage": avg_cov,
                "mae": mae,
                "rmse": rmse,
                "pinball_loss": pinball,
                "groupwise_pinball_loss": grp_pinball,
            },
            "baseline": {
                "avg_groupwise_coverage": baseline_cov,
                "mae": baseline_mae,
                "rmse": baseline_rmse,
                "pinball_loss": baseline_pinball,
                "groupwise_pinball_loss": baseline_grp_pinball,
            },
        },
        "sample_predictions": samples,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_content, f, indent=2)
    print(f"[INFO] Logged → {log_path}")

    # Optionally save trained booster
    save_dir = os.getenv("SAVED_MODELS")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_model(os.path.join(save_dir, f"{model_name}.json"))


# ---------------------------------------------------------------------------
# Run both quantiles
# ---------------------------------------------------------------------------
def train_all_quantiles():
    train_boosted_model(0.9, "xgb_quantile_90")
    train_boosted_model(0.5, "xgb_quantile_50")


if __name__ == "__main__":
    train_all_quantiles()
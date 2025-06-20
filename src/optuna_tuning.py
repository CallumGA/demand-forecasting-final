#!/usr/bin/env python
"""
Optuna hyper-parameter search for 90-th & 50-th quantile XGBoost.
Prints the best params only – no MLflow, no final full-data training.

Compatible with XGBoost 2.x (eval_metric moved to constructor).
"""

# ── Paths ───────────────────────────────────────────────────────
import os
os.environ.setdefault(
    "CLEANED_SALES_DATA",
    "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/sales_cleaned.csv"
)

# ── Search settings ─────────────────────────────────────────────
N_TRIALS      = 50          # per quantile
PARALLEL_JOBS = 4           # concurrent trials (≤ CPU cores)
SAMPLE_FRAC   = 0.10        # ~10 % of rows per trial
VAL_DAYS      = 28          # hold-out window
EARLY_STOP    = 30          # XGB early-stopping rounds
QUANTILES     = [0.90, 0.50]
FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id"
]
# ────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import optuna
import xgboost
from functools import partial

# ── Helpers ─────────────────────────────────────────────────────
def groupwise_time_split(df, val_days=VAL_DAYS):
    train, val = [], []
    for _, g in df.groupby(["item_id", "store_id"]):
        g = g.sort_values("d")
        if len(g) > val_days + 30:
            train.append(g.iloc[:-val_days])
            val.append(g.iloc[-val_days:])
    return pd.concat(train), pd.concat(val)

def pinball(y_true, y_pred, q):
    d = y_true - y_pred
    return np.mean(np.maximum(q * d, (q - 1) * d))

def groupwise_pinball(df, q):
    return (df.groupby(["item_id", "store_id"])
              .apply(lambda g: pinball(g["sales"], g["pred"], q))
              .mean())

# ── One-time data load ─────────────────────────────────────────
df_full = pd.read_csv(os.getenv("CLEANED_SALES_DATA"))
df_full["item_id"]  = df_full["item_id"].astype("category")
df_full["store_id"] = df_full["store_id"].astype("category")
train_df, val_df = groupwise_time_split(df_full)
X_val, y_val = val_df[FEATURES], val_df["sales"]

# ── Optuna objective ───────────────────────────────────────────
def objective(q: float, trial: optuna.trial.Trial):
    # trial-specific subsample
    mask = np.random.rand(len(train_df)) < SAMPLE_FRAC
    X_sub = train_df.loc[mask, FEATURES]
    y_sub = train_df.loc[mask, "sales"]

    params = {
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("lr", 0.01, 0.3, log=True),
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
        "gamma":            trial.suggest_float("gamma", 0, 5),
        "reg_lambda":       trial.suggest_float("lambda_l2", 0, 5),
        "reg_alpha":        trial.suggest_float("lambda_l1", 0, 5),
        # fixed
        "n_jobs": 1,
        "tree_method": "hist",
        "objective": "reg:quantileerror",
        "quantile_alpha": q,
        "eval_metric": "quantile",     # ← moved here (works on 2.x)
        "enable_categorical": True,
        "early_stopping_rounds": EARLY_STOP,
        "random_state": trial.number,
    }

    model = xgboost.XGBRegressor(**params)
    model.fit(
        X_sub, y_sub,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)
    loss = groupwise_pinball(val_df.assign(pred=preds), q)
    return loss

# ── Run search per quantile ────────────────────────────────────
def run_search(q):
    print(f"\n=== Searching {int(q*100)}th-quantile ===")
    pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=10, reduction_factor=4)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(partial(objective, q), n_trials=N_TRIALS, n_jobs=PARALLEL_JOBS)

    print(f"Best group-pinball loss: {study.best_value:.5f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("-" * 40)
    return study.best_params

if __name__ == "__main__":
    best_by_quantile = {int(q*100): run_search(q) for q in QUANTILES}

    # Pretty summary to copy into your training script
    print("\n===== COPY THESE INTO YOUR CODE =====")
    for q, params in best_by_quantile.items():
        print(f"{q}th quantile → {params}")
import os, csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

"""
****************************************************************
DRF Point-Forecast – Predictions + Metrics  (no intervals)
Run standalone after randomforest_model_training.py
****************************************************************
"""


# ──────────────────────────────────────────────────────────────
# 1. Config & paths
# ──────────────────────────────────────────────────────────────
FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id", "is_weekend"
]
WINDOW_BASELINE = 28   # rolling-mean window for naïve baseline

ROOT  = "/Users/callumanderson/Documents/Documents - Callum’s Laptop/MIA5130/final-project/final-project-implementation"
MODEL = os.path.join(ROOT, "models", "DRF_model_python_1750806434317_1")

DATA_P    = os.path.expanduser("~/h2o_data")
EVAL_CSV  = os.path.join(DATA_P, "evaluation_input_data.csv")
TRAIN_CSV = os.path.join(DATA_P, "training_input_data.csv")
TRUTH_CSV = os.path.join(DATA_P, "sales_train_evaluation.csv")

OUT_DIR = os.path.join(DATA_P, "real_evaluation_predictions")
os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# 2. Helper – vectorised 28-day baseline
# ──────────────────────────────────────────────────────────────
def baseline_28d(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                 window: int = WINDOW_BASELINE) -> np.ndarray:
    train_df = train_df.sort_values(["item_id", "store_id", "d"])
    bl = (train_df
          .groupby(["item_id", "store_id"], observed=True)["sales"]
          .apply(lambda s: s.iloc[-window:].mean() if len(s) >= window else s.mean())
          .rename("baseline_pred")
          .reset_index())
    return (eval_df
            .merge(bl, on=["item_id", "store_id"], how="left")["baseline_pred"]
            .to_numpy())


# ──────────────────────────────────────────────────────────────
# 3. Load model & data
# ──────────────────────────────────────────────────────────────
h2o.init(max_mem_size="6G", nthreads=-1)
model = h2o.load_model("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/models/DRF_model_python_1750806434317_1")
print("Model loaded.")

eval_df  = pd.read_csv(EVAL_CSV)
train_df = pd.read_csv(TRAIN_CSV)

for df in (eval_df, train_df):
    df["item_id"]  = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

# Point-forecasts
eval_df["predicted_sales"] = (
    model.predict(h2o.H2OFrame(eval_df[FEATURES]))
         .as_data_frame()["predict"]
         .astype(np.float32)
)
print("Forecasts generated.")

# Ground-truth melt & merge
truth_df = (
    pd.read_csv(TRUTH_CSV)
      .melt(id_vars=["id", "item_id", "dept_id", "cat_id",
                     "store_id", "state_id"],
            var_name="d",
            value_name="actual_sales")
)
merged = (eval_df
          .merge(truth_df[["item_id", "store_id", "d", "actual_sales"]],
                 on=["item_id", "store_id", "d"],
                 how="left")
          .dropna(subset=["actual_sales"])
)
print("🔗  Ground-truth merged –", len(merged), "rows.")


# ───── 4. Baseline & metrics ─────
merged["baseline_pred"] = baseline_28d(train_df, merged)

mae  = mean_absolute_error(merged["actual_sales"], merged["predicted_sales"])
rmse = np.sqrt( mean_squared_error(merged["actual_sales"],
                                   merged["predicted_sales"]) )

baseline_mae  = mean_absolute_error(merged["actual_sales"], merged["baseline_pred"])
baseline_rmse = np.sqrt( mean_squared_error(merged["actual_sales"],
                                            merged["baseline_pred"]) )

print(f"\n───────── METRICS ─────────")
print(f"Baseline  MAE  {baseline_mae:8.4f}   RMSE {baseline_rmse:8.4f}")
print(f"DRF       MAE  {mae:8.4f}   RMSE {rmse:8.4f}")
print("───────────────────────────")


# ──────────────────────────────────────────────────────────────
# 5. Save CSVs
# ──────────────────────────────────────────────────────────────
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# 5·1  Metrics summary
summary_csv = os.path.join(OUT_DIR,
    f"drf_point_forecast_eval_metrics_{ts}.csv")
with open(summary_csv, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=[
        "model_name", "mae", "rmse",
        "baseline_mae", "baseline_rmse", "timestamp"])
    writer.writeheader()
    writer.writerow({
        "model_name": "drf_point_forecast",
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "timestamp": ts
    })
print("Metrics saved →", summary_csv)

# 5·2  Full per-row predictions
pred_csv = os.path.join(OUT_DIR,
    f"drf_point_forecast_eval_predictions_full_{ts}.csv")
merged[["item_id", "store_id", "d",
        "actual_sales", "predicted_sales", "baseline_pred"]
      ].to_csv(pred_csv, index=False)
print("Full predictions saved →", pred_csv)

# ──────────────────────────────────────────────────────────────
# 6. Shutdown
# ──────────────────────────────────────────────────────────────
h2o.cluster().shutdown(prompt=False)
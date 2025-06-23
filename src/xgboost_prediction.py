import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from datetime import datetime
import csv

"""
****************************************************************
 Point-Forecast w/ Groupwise Interval Model Predictions
 Note: Run standalone after xgboost_model_training.py
****************************************************************
"""


# the features xgboost trees will branch on
FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id"
]

CONFIDENCE_LEVEL = 0.95
CALIB_FRAC = 0.10

# load the model we trained and validated with a rough 90/10 split evenly accross product/location pairs
model_path = "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/models/xgb_point_forecast.joblib"
model = joblib.load(model_path)

# load evaluation input
df_eval = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/eval_prediction_input.csv")
df_eval["item_id"] = df_eval["item_id"].astype("category")
df_eval["store_id"] = df_eval["store_id"].astype("category")
X_eval = df_eval[FEATURES]
df_eval["predicted_sales"] = model.predict(X_eval).astype(np.float32)

# load actual sales
df_truth = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/raw/sales_train_evaluation.csv")
df_truth = df_truth.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="actual_sales"
)

# merge evaluation dataframe with actual sales
df_merged = pd.merge(
    df_eval,
    df_truth[["item_id", "store_id", "d", "actual_sales"]],
    on=["item_id", "store_id", "d"],
    how="left"
).dropna(subset=["actual_sales"])

# load training input
df_train = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/sales_cleaned.csv")
df_train["item_id"] = df_train["item_id"].astype("category")
df_train["store_id"] = df_train["store_id"].astype("category")
X_train = df_train[FEATURES]
y_train = df_train["sales"]


# *** Add prediction bands for each product/location group ***
#  We use the "error" from the training model for each product/location pair and + or - from the new prediction to get the interval band for new predictions
#  Compute the ε:
#       abs_res = (yi - ŷi), where yi = actual sales from training data (per item/location), ŷi = predicted sales from training data (per item/location)
#       ε = np.quantile(abs_res, 1 - alpha / 2)
#  Derive the lower and upper bounds from ε:
#       Lower = max(0, ŷ_new − ε), Upper = (ŷ_new + ε), where ŷ_new = new predicted sales,  ε = epsilon
def add_groupwise_prediction_intervals_final(model, df_eval, df_train, X_train, y_train,
                                             confidence_level=0.95, calib_frac=0.10):
    df_eval = df_eval.copy()
    lower = np.zeros(len(df_eval))
    upper = np.zeros(len(df_eval))
    interval_width = np.zeros(len(df_eval))
    alpha = 1 - confidence_level

    for key, group in df_eval.groupby(["item_id", "store_id"], observed=True):
        mask = (df_train["item_id"] == key[0]) & (df_train["store_id"] == key[1])
        X_tg = X_train[mask]
        y_tg = y_train[mask]

        if len(y_tg) < 10:
            eps = 0.0
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(X_tg), max(1, int(calib_frac * len(X_tg))), replace=False)
            abs_res = np.abs(y_tg.iloc[idx] - model.predict(X_tg.iloc[idx]))
            eps = np.quantile(abs_res, 1 - alpha / 2)

        idx_eval = df_eval[(df_eval["item_id"] == key[0]) & (df_eval["store_id"] == key[1])].index
        lower[idx_eval] = np.maximum(0, df_eval.loc[idx_eval, "predicted_sales"] - eps)
        upper[idx_eval] = df_eval.loc[idx_eval, "predicted_sales"] + eps
        interval_width[idx_eval] = upper[idx_eval] - lower[idx_eval]

    return {
        "lower_bound": lower,
        "upper_bound": upper,
        "interval_width": interval_width,
        "confidence_level": confidence_level
    }

intervals = add_groupwise_prediction_intervals_final(
    model, df_eval, df_train, X_train, y_train,
    confidence_level=CONFIDENCE_LEVEL, calib_frac=CALIB_FRAC
)

# add to merged df
df_merged["lower_bound"] = intervals["lower_bound"]
df_merged["upper_bound"] = intervals["upper_bound"]

# evaluate the forecasts made
mae = mean_absolute_error(df_merged["actual_sales"], df_merged["predicted_sales"])
rmse = np.sqrt(mean_squared_error(df_merged["actual_sales"], df_merged["predicted_sales"]))


# *** Evaluate our interval bands to see how well they performed for evaluation ***
def evaluate_prediction_intervals(y_true, intv):
    within = (y_true >= intv["lower_bound"]) & (y_true <= intv["upper_bound"])
    return {
        "coverage": float(within.mean()),
        "average_width": float(intv["interval_width"].mean()),
        "expected_coverage": intv["confidence_level"]
    }

interval_metrics = evaluate_prediction_intervals(
    df_merged["actual_sales"],
    {
        "lower_bound": df_merged["lower_bound"],
        "upper_bound": df_merged["upper_bound"],
        "interval_width": df_merged["upper_bound"] - df_merged["lower_bound"],
        "confidence_level": CONFIDENCE_LEVEL
    }
)

# *** Compute the evaluation baseline to compare our predictions against ***
def compute_eval_baseline_predictions(train_df, eval_df, window=28):
    preds = []
    for key, g_eval in eval_df.groupby(["item_id", "store_id"], observed=True):
        g_train = train_df.loc[
            (train_df["item_id"] == key[0]) & (train_df["store_id"] == key[1])
        ].sort_values("d")

        if len(g_train) < window:
            baseline = g_train["sales"].mean()
        else:
            baseline = g_train["sales"].rolling(window=window, min_periods=window).mean().iloc[-1]
        preds.extend([baseline] * len(g_eval))
    return np.array(preds)

df_merged["baseline_pred"] = compute_eval_baseline_predictions(df_train, df_merged)
baseline_mae = mean_absolute_error(df_merged["actual_sales"], df_merged["baseline_pred"])
baseline_rmse = np.sqrt(mean_squared_error(df_merged["actual_sales"], df_merged["baseline_pred"]))

# output dir setup
log_dir = "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/real_evaluation_predictions/"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# save result metric summary csv
summary_path = os.path.join(log_dir, f"xgb_point_forecast_eval_metrics_{timestamp}.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model_name", "mae", "rmse", "baseline_mae", "baseline_rmse",
        "interval_coverage", "interval_avg_width", "expected_coverage", "timestamp"
    ])
    writer.writeheader()
    writer.writerow({
        "model_name": "xgb_point_forecast",
        "mae": mae,
        "rmse": rmse,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "interval_coverage": interval_metrics["coverage"],
        "interval_avg_width": interval_metrics["average_width"],
        "expected_coverage": interval_metrics["expected_coverage"],
        "timestamp": timestamp
    })

# save 100 of our sample predictions for evaluation
sample_path = os.path.join(log_dir, f"xgb_point_forecast_eval_predictions_{timestamp}.csv")
sample_df = df_merged.sample(100, random_state=42)
sample_df = sample_df[[
    "item_id", "store_id", "d", "actual_sales", "predicted_sales",
    "baseline_pred", "lower_bound", "upper_bound"
]]
sample_df["in_band"] = (
    (sample_df["actual_sales"] >= sample_df["lower_bound"]) &
    (sample_df["actual_sales"] <= sample_df["upper_bound"])
)
sample_df.to_csv(sample_path, index=False)

# save the full prediction results
full_path = os.path.join(log_dir, f"xgb_point_forecast_eval_predictions_full_{timestamp}.csv")
full_df = df_merged[[
    "item_id", "store_id", "d", "actual_sales", "predicted_sales",
    "baseline_pred", "lower_bound", "upper_bound"
]].copy()
full_df["in_band"] = (
    (full_df["actual_sales"] >= full_df["lower_bound"]) &
    (full_df["actual_sales"] <= full_df["upper_bound"])
)
full_df.to_csv(full_path, index=False)
print(f"📦 Full predictions saved: {full_path}")

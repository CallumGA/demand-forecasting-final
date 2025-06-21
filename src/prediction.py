import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the pre-trained model
model_path = "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/models/xgb_point_forecast.joblib"
model = joblib.load(model_path)

# Read the prediction input
df_eval = pd.read_csv(
    "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/eval_prediction_input.csv"
)

FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id"
]

X_eval = df_eval[FEATURES].copy()
X_eval["item_id"] = X_eval["item_id"].astype("category")
X_eval["store_id"] = X_eval["store_id"].astype("category")

# Predict point forecasts
df_eval["predicted_sales"] = model.predict(X_eval).astype(np.float32)

# Load the actual sales for truth values
df_truth = pd.read_csv(
    "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/raw/sales_train_evaluation.csv"
)

df_truth = df_truth.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="actual_sales"
)

df_merged = pd.merge(
    df_eval,
    df_truth[["item_id", "store_id", "d", "actual_sales"]],
    on=["item_id", "store_id", "d"],
    how="left"
)

df_merged = df_merged.dropna(subset=["actual_sales"])

# Load the original training data for baseline and interval calibration
df_train = pd.read_csv(
    "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/sales_cleaned.csv"
)
df_train["item_id"] = df_train["item_id"].astype("category")
df_train["store_id"] = df_train["store_id"].astype("category")
X_train = df_train[FEATURES]
y_train = df_train["sales"]

# Create the prediction intervals
def add_prediction_intervals(model, X_val, X_train, y_train,
                             confidence_level=0.95, calib_frac=0.10):
    point = model.predict(X_val)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_train), int(calib_frac * len(X_train)), replace=False)
    abs_res = np.abs(y_train.iloc[idx] - model.predict(X_train.iloc[idx]))

    alpha = 1 - confidence_level
    eps = np.quantile(abs_res, 1 - alpha / 2)

    lower = np.maximum(0, point - eps)
    upper = point + eps

    return {
        "point_forecast": point,
        "lower_bound": lower,
        "upper_bound": upper,
        "interval_width": upper - lower,
        "epsilon": eps,
        "confidence_level": confidence_level,
    }

def evaluate_prediction_intervals(y_true, intv):
    within = (y_true >= intv["lower_bound"]) & (y_true <= intv["upper_bound"])
    return {
        "coverage": float(within.mean()),
        "average_width": float(intv["interval_width"].mean()),
        "expected_coverage": intv["confidence_level"],
    }

intervals = add_prediction_intervals(
    model, X_eval, X_train, y_train, confidence_level=0.95
)

df_merged["lower_bound"] = intervals["lower_bound"]
df_merged["upper_bound"] = intervals["upper_bound"]

# Evaluate model performance
mae = mean_absolute_error(df_merged["actual_sales"], df_merged["predicted_sales"])
rmse = np.sqrt(mean_squared_error(df_merged["actual_sales"], df_merged["predicted_sales"]))

interval_metrics = evaluate_prediction_intervals(
    df_merged["actual_sales"],
    {
        "lower_bound": df_merged["lower_bound"],
        "upper_bound": df_merged["upper_bound"],
        "interval_width": df_merged["upper_bound"] - df_merged["lower_bound"],
        "confidence_level": 0.95,
    }
)

# ⛳️ Compute Baseline Predictions on Evaluation Set
def compute_eval_baseline_predictions(train_df, eval_df, window=28):
    preds = []
    for key, g_eval in eval_df.groupby(["item_id", "store_id"], observed=True):
        g_train = train_df.loc[
            (train_df["item_id"] == key[0]) & (train_df["store_id"] == key[1])
        ].sort_values("d")

        if len(g_train) < window:
            baseline = g_train["sales"].mean()
        else:
            baseline = (
                g_train["sales"]
                .rolling(window=window, min_periods=window)
                .mean()
                .iloc[-1]
            )
        preds.extend([baseline] * len(g_eval))
    return np.array(preds)

df_merged["baseline_pred"] = compute_eval_baseline_predictions(df_train, df_merged)

baseline_mae = mean_absolute_error(df_merged["actual_sales"], df_merged["baseline_pred"])
baseline_rmse = np.sqrt(mean_squared_error(df_merged["actual_sales"], df_merged["baseline_pred"]))

# Save results
output_path = "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/eval_predictions_with_truth.csv"
df_merged.to_csv(output_path, index=False)

# Final output
print(f"✓ Predictions with actuals and intervals saved to: {output_path}")
print(f"📊 Final MAE: {mae:.4f} | RMSE: {rmse:.4f}")
print(f"🎯 95% Interval Coverage: {interval_metrics['coverage']:.4f}")
print(f"📏 Avg Interval Width: {interval_metrics['average_width']:.4f}")
print(f"🧪 Baseline MAE: {baseline_mae:.4f} | Baseline RMSE: {baseline_rmse:.4f}")
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

# TODO: restructure/organize code to be more readable, then train/predict models, then finalize and add to submission package


# the features xgboost trees will branch on
FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id", "is_weekend"
]

CONFIDENCE_LEVEL = 0.95
CALIB_FRAC = 0.10

# load the model we trained and validated with a rough 90/10 split evenly across product/location pairs
model_path = "/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/models/xgb_point_forecast.joblib"
trained_model = joblib.load(model_path)

# load evaluation input
evaluation_data_df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/evaluation_input_data.csv")
evaluation_data_df["item_id"] = evaluation_data_df["item_id"].astype("category")
evaluation_data_df["store_id"] = evaluation_data_df["store_id"].astype("category")

# make our predictions with our input features
X_evaluation = evaluation_data_df[FEATURES]
evaluation_data_df["predicted_sales"] = trained_model.predict(X_evaluation).astype(np.float32)

# load actual sales
true_predictions_df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/raw/sales_train_evaluation.csv")
true_predictions_df = true_predictions_df.melt(
    id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
    var_name="d",
    value_name="actual_sales"
)

# merge evaluation dataframe with actual sales
merged_df = pd.merge(
    evaluation_data_df,
    true_predictions_df[["item_id", "store_id", "d", "actual_sales"]],
    on=["item_id", "store_id", "d"],
    how="left"
).dropna(subset=["actual_sales"])

# load training input
train_data_df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/training_input_data.csv")
train_data_df["item_id"] = train_data_df["item_id"].astype("category")
train_data_df["store_id"] = train_data_df["store_id"].astype("category")
X_train = train_data_df[FEATURES]
y_train = train_data_df["sales"]


# *** Add prediction bands for each product/location group ***
#  We use the "error" from the training model for each product/location pair and + or - from the new prediction to get the interval band for new predictions
#  Compute the ε:
#       abs_res = (yi - ŷi), where yi = actual sales from training data (per item/location), ŷi = predicted sales from training data (per item/location)
#       ε = np.quantile(abs_res, 1 - alpha / 2)
#  Derive the lower and upper bounds from ε:
#       Lower = max(0, ŷ_new − ε), Upper = (ŷ_new + ε), where ŷ_new = new predicted sales,  ε = epsilon
def add_groupwise_prediction_intervals_final(model, evaluation_df, df_train, X_train, y_train,
                                             confidence_level=0.95, calib_frac=0.10):
    evaluation_df = evaluation_df.copy()
    lower_bound = np.zeros(len(evaluation_df))
    upper_bound = np.zeros(len(evaluation_df))
    interval_width = np.zeros(len(evaluation_df))
    alpha = 1 - confidence_level

    for key, group in evaluation_df.groupby(["item_id", "store_id"], observed=True):
        mask = (df_train["item_id"] == key[0]) & (df_train["store_id"] == key[1])
        X_training_features = X_train[mask]
        y_sales = y_train[mask]

        if len(y_sales) < 10:
            epsilon = 0.0
        else:
            range = np.random.default_rng(42)
            idx = range.choice(len(X_training_features), max(1, int(calib_frac * len(X_training_features))), replace=False)
            absolute_residuals = np.abs(y_sales.iloc[idx] - model.predict(X_training_features.iloc[idx]))
            epsilon = np.quantile(absolute_residuals, 1 - alpha / 2)

        idx_eval = evaluation_df[(evaluation_df["item_id"] == key[0]) & (evaluation_df["store_id"] == key[1])].index
        lower_bound[idx_eval] = np.maximum(0, evaluation_df.loc[idx_eval, "predicted_sales"] - epsilon)
        upper_bound[idx_eval] = evaluation_df.loc[idx_eval, "predicted_sales"] + epsilon
        interval_width[idx_eval] = upper_bound[idx_eval] - lower_bound[idx_eval]

    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "interval_width": interval_width,
        "confidence_level": confidence_level
    }

intervals = add_groupwise_prediction_intervals_final(
    trained_model, evaluation_data_df, train_data_df, X_train, y_train,
    confidence_level=CONFIDENCE_LEVEL, calib_frac=CALIB_FRAC
)

# add to merged df
merged_df["lower_bound"] = intervals["lower_bound"]
merged_df["upper_bound"] = intervals["upper_bound"]

# calculate MAE and RMSE against actual sales for actual final predictions
mae = mean_absolute_error(merged_df["actual_sales"], merged_df["predicted_sales"])
rmse = np.sqrt(mean_squared_error(merged_df["actual_sales"], merged_df["predicted_sales"]))


# *** Evaluate our interval bands to see how well they performed for evaluation ***
def evaluate_prediction_intervals(y_true, interval):
    within_bounds = (y_true >= interval["lower_bound"]) & (y_true <= interval["upper_bound"])
    return {
        "coverage": float(within_bounds.mean()),
        "average_width": float(interval["interval_width"].mean()),
        "expected_coverage": interval["confidence_level"]
    }

interval_metrics = evaluate_prediction_intervals(
    merged_df["actual_sales"],
    {
        "lower_bound": merged_df["lower_bound"],
        "upper_bound": merged_df["upper_bound"],
        "interval_width": merged_df["upper_bound"] - merged_df["lower_bound"],
        "confidence_level": CONFIDENCE_LEVEL
    }
)

# *** Compute the evaluation baseline to compare our predictions against ***
def compute_eval_baseline_predictions(train_df, evaluation_df, window=28):
    baseline_predictions = []
    for key, group in evaluation_df.groupby(["item_id", "store_id"], observed=True):
        g_train = train_df.loc[
            (train_df["item_id"] == key[0]) & (train_df["store_id"] == key[1])
        ].sort_values("d")

        if len(g_train) < window:
            baseline = g_train["sales"].mean()
        else:
            baseline = g_train["sales"].rolling(window=window, min_periods=window).mean().iloc[-1]
        baseline_predictions.extend([baseline] * len(group))
    return np.array(baseline_predictions)

# calculate MAE and RMSE against actual sales for baseline predictions
merged_df["baseline_pred"] = compute_eval_baseline_predictions(train_data_df, merged_df)
baseline_mae = mean_absolute_error(merged_df["actual_sales"], merged_df["baseline_pred"])
baseline_rmse = np.sqrt(mean_squared_error(merged_df["actual_sales"], merged_df["baseline_pred"]))

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


# save the full prediction results
full_path = os.path.join(log_dir, f"xgb_point_forecast_eval_predictions_full_{timestamp}.csv")
full_df = merged_df[[
    "item_id", "store_id", "d", "actual_sales", "predicted_sales",
    "baseline_pred", "lower_bound", "upper_bound"
]].copy()
full_df["in_band"] = (
    (full_df["actual_sales"] >= full_df["lower_bound"]) &
    (full_df["actual_sales"] <= full_df["upper_bound"])
)
full_df.to_csv(full_path, index=False)
print(f"Full predictions saved: {full_path}")

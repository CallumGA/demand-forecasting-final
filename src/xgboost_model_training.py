import os
import pandas as pd
import xgboost
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models.signature import infer_signature

"""
    ****************************************************************
     Train and log quantile XGBoost models via MLflow.
    ****************************************************************
    Run in Terminal: mlflow ui
    Visit http://127.0.0.1:5000 after running `mlflow ui` to view results.

    1. Split per group — the last 28 days of each item/store pair.
    2. Train and evaluate globally, but preserve group info for evaluation.
    3. Measure calibration and pinball loss per group to detect problems like under-predicting spikes.
"""

# TODO: Log updated to handle logging PER product id/store location
# TODO: Everything needs to be done PER product id/store location or else the model won't learn properly
# TODO: The 90/10 split must be PER product id/store location

# set URI for mlflow evaluation
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# centralized configuration for hyperparameters per quantile
QUANTILE_CONFIGS = {
    0.9: {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8},
    # max-depth: lower 6->4 for 50th qunatile. Will smooth out results/predictions (pay less attention to spikes)
    # learning-rate: lower from 1 to 0.05. Helps model learn more gradually. Focus more on stable patterns.
    # n_estimators: 200->300 Balances more trees paired with smaller learning rate. Helps avoid settling on conservative trends too soon.
    # subsample: 0.8-<0.9. Encourages diversity in the trees (less chance of uniform bias)
    # colsample_bytree: 0.8->0.9. Helps avoid over-reliance on a small number of features (e.g., price or lag)
    0.5: {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.9, "colsample_bytree": 0.9}
}

def groupwise_time_split(df, val_days=28):
    train_list, val_list = [], []
    for _, group in df.groupby(["item_id", "store_id"]):
        group = group.sort_values("d")
        if len(group) > val_days + 30:
            train_list.append(group.iloc[:-val_days])
            val_list.append(group.iloc[-val_days:])
    return pd.concat(train_list), pd.concat(val_list)

def train_boosted_model(quantile_alpha: float, model_name: str):
    # read cleaned data csv
    df = pd.read_csv(os.getenv("CLEANED_SALES_DATA"))

    # make the non-numerical fields categorical for xgboost
    df["item_id"] = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    # define features
    features = [
        "sell_price", "snap", "is_event_day", "lag_7", "rolling_mean_7",
        "day_of_week", "price_change_pct", "month", "item_id", "store_id"
    ]

    # do a time-based 90/10 split PER product/store location
    train_df, val_df = groupwise_time_split(df)

    # extract training and validation sets
    X_train = train_df[features]
    y_train = train_df["sales"]
    X_val = val_df[features]
    y_val = val_df["sales"]

    # get the correct hyperparams for the quantile we want
    config = QUANTILE_CONFIGS[quantile_alpha]

    # apply the hyperparameters
    model = xgboost.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=quantile_alpha,
        tree_method="hist",
        enable_categorical=True,
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        n_estimators=config["n_estimators"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        random_state=42,
        early_stopping_rounds=30,
    )

    # start the mlflow logging for evaluation
    with mlflow.start_run(run_name=f"Quantile_{int(quantile_alpha * 100)}"):
        mlflow.log_params({"quantile_alpha": quantile_alpha, **config})

        # fit the model - this is where we train and validate
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # make our actual predictions
        y_pred = model.predict(X_val)

        # attach predictions back to the val_df
        val_df = val_df.copy()
        val_df["pred"] = y_pred
        val_df["covered"] = (val_df["sales"] <= y_pred).astype(int)

        # calculate groupwise coverage score (calibration) per product/store
        group_eval = val_df.groupby(["item_id", "store_id"]).agg({
            "sales": "mean",
            "pred": "mean",
            "covered": "mean"
        }).reset_index()

        # metric 1: Calibration Score (average groupwise coverage)
        avg_coverage = group_eval["covered"].mean()

        # metric 2: MAE (classical error metric)
        mae = mean_absolute_error(y_val, y_pred)

        # metric 3: RMSE (classical error metric)
        rmse = mean_squared_error(y_val, y_pred) ** 0.5

        # log all metrics to mlflow
        mlflow.log_metric("avg_groupwise_coverage", avg_coverage)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)

        # save model to disk + log to MLflow
        save_path = os.getenv("SAVED_MODELS")
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save_model(os.path.join(save_path, f"{model_name}.json"))

        # log model with schema and input example
        signature = infer_signature(X_val, y_pred)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            input_example=X_val.iloc[:1],
            signature=signature
        )

def train_all_quantiles():
    train_boosted_model(quantile_alpha=0.9, model_name="xgb_quantile_90")
    train_boosted_model(quantile_alpha=0.5, model_name="xgb_quantile_50")

if __name__ == "__main__":
    train_all_quantiles()
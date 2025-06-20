import os
import numpy as np
import pandas as pd
import xgboost
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
from mlflow.models.signature import infer_signature
import optuna

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

# TODO: Add hyperparams from the optuna outputs.....
# TODO: Remove optuna and ml flow when model is performing properlyExample Workflow:
# TODO: Build a 28-day rolling-quantile baseline for each item × store, compute its group-wise pinball loss and coverage using the same groupwise_pinball_loss function as the model, and log both baseline and model metrics side-by-side for direct comparison.
# TODO: Add SHAP to explain features and choose the best ones
# 	1.	Run SHAP on your current model
# 	2.	Identify underperforming regions (bad predictions)
# 	3.	See which features are driving those errors
# 	4.	Hypothesize new engineered features (e.g. lags, interactions, log transforms)
# 	5.	Retrain and re-SHAP — repeat!

# set URI for mlflow evaluation
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# centralized configuration for hyperparameters per quantile (if not tuning)
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
        "reg_alpha": 1.0946520339010668
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
        "reg_alpha": 3.184399463923405
    }
}

def groupwise_time_split(df, val_days=28):
    train_list, val_list = [], []
    for _, group in df.groupby(["item_id", "store_id"]):
        group = group.sort_values("d")
        if len(group) > val_days + 30:
            train_list.append(group.iloc[:-val_days])
            val_list.append(group.iloc[-val_days:])
    return pd.concat(train_list), pd.concat(val_list)


def compute_pinball_loss(y_true, y_pred, quantile):
    delta = y_true - y_pred
    return np.mean(np.maximum(quantile * delta, (quantile - 1) * delta))


def groupwise_pinball_loss(df, quantile):
    def pinball(y_true, y_pred):
        delta = y_true - y_pred
        return np.mean(np.maximum(quantile * delta, (quantile - 1) * delta))

    grouped = df.groupby(["item_id", "store_id"])
    losses = grouped.apply(lambda g: pinball(g["sales"], g["pred"]))
    return losses.mean()


def train_boosted_model(quantile_alpha: float, model_name: str, use_optuna=False):
    df = pd.read_csv(os.getenv("CLEANED_SALES_DATA"))
    df["item_id"] = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    features = [
        "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
        "day_of_week", "month", "item_id", "store_id"
    ]

    train_df, val_df = groupwise_time_split(df)
    X_train, y_train = train_df[features], train_df["sales"]
    X_val, y_val = val_df[features], val_df["sales"]

    def objective(trial):
        config = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = xgboost.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=quantile_alpha,
            tree_method="hist",
            enable_categorical=True,
            random_state=42,
            early_stopping_rounds=30,
            **config,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        # compute groupwise pinball loss for fair evaluation
        val_copy = val_df.copy()
        val_copy["pred"] = y_pred
        return groupwise_pinball_loss(val_copy, quantile_alpha)

    if use_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, n_jobs=5)
        config = study.best_params
    else:
        config = QUANTILE_CONFIGS[quantile_alpha]

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

    with mlflow.start_run(run_name=f"Quantile_{int(quantile_alpha * 100)}"):
        mlflow.log_params({"quantile_alpha": quantile_alpha, **config})
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        val_df = val_df.copy()
        val_df["pred"] = y_pred
        val_df["covered"] = (val_df["sales"] <= y_pred).astype(int)

        group_eval = val_df.groupby(["item_id", "store_id"]).agg({
            "sales": "mean", "pred": "mean", "covered": "mean"
        }).reset_index()

        avg_coverage = group_eval["covered"].mean()
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        pinball = compute_pinball_loss(y_val, y_pred, quantile_alpha)
        group_pinball = groupwise_pinball_loss(val_df, quantile_alpha)

        mlflow.log_metric("avg_groupwise_coverage", avg_coverage)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("pinball_loss", pinball)
        mlflow.log_metric("groupwise_pinball_loss", group_pinball)

        sample_df = val_df.groupby(["item_id", "store_id"]).apply(
            lambda g: g.sample(1, random_state=42)).reset_index(drop=True)
        mlflow.log_text(sample_df[["item_id", "store_id", "sales", "pred"]].to_csv(index=False),
                        "sample_predictions.csv")

        save_path = os.getenv("SAVED_MODELS")
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save_model(os.path.join(save_path, f"{model_name}.json"))

        signature = infer_signature(X_val, y_pred)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            input_example=X_val.iloc[:1],
            signature=signature
        )


def train_all_quantiles():
    train_boosted_model(quantile_alpha=0.9, model_name="xgb_quantile_90", use_optuna=True)
    train_boosted_model(quantile_alpha=0.5, model_name="xgb_quantile_50", use_optuna=True)


if __name__ == "__main__":
    train_all_quantiles()
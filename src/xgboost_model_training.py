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
"""

# set URI for mlflow evaluation
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# centralized configuration for hyperparameters per quantile
QUANTILE_CONFIGS = {
    0.9: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    0.5: {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
}


def train_boosted_model(quantile_alpha: float, model_name: str):

    # read cleaned data csv
    df = pd.read_csv(os.getenv("CLEANED_SALES_DATA"))

    # make the non-numerical fields categorical for xgboost
    df["item_id"] = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    # split data into y target (sales) and x (features)
    y = df["sales"]
    X = df[[
        "sell_price", "snap", "is_event_day", "lag_7", "rolling_mean_7",
        "day_of_week", "price_change_pct", "month", "item_id", "store_id"
    ]]

    """
    X_train: First 90% of feature data used to train the model
    y_train: First 90% of sales values corresponding to X_train

    X_val: Last 10% of feature data used as input to the model for prediction
    y_val: Last 10% of actual sales values, used to compare against predictions

    This setup simulates forecasting future sales using past data, 
    where we train on historical patterns and validate predictions 
    on unseen, more recent data — helping evaluate model performance realistically.
    """
    val_size = int(0.1 * len(X))
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]

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

    # start the ml flow logging for evaluation
    with mlflow.start_run(run_name=f"Quantile_{int(quantile_alpha * 100)}"):
        mlflow.log_params({
            "quantile_alpha": quantile_alpha,
            **config
        })

        # fit the model - this is where we train and validate
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # make our actual predictions
        y_pred = model.predict(X_val)

        # construct results logging objects
        val_context = df.iloc[-val_size:][["item_id", "store_id", "d"]].reset_index(drop=True)
        results = pd.DataFrame({
            "item_id": val_context["item_id"],
            "store_id": val_context["store_id"],
            "d": val_context["d"],
            "Actual Sales": y_val.values,
            "Predicted Threshold": y_pred
        })

        # metric 1: Calibration Score (custom quantile coverage)
        calibration_score = (results["Actual Sales"] <= results["Predicted Threshold"]).mean()
        mlflow.log_metric("calibration_score", calibration_score)

        # metric 2 & 3: MAE and RMSE (classical error metrics)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
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
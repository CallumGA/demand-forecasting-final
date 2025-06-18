import os
import pandas as pd
import xgboost

"""
    ****************************************************************
     Train and optimize the model including hyper-parameter tuning.
    ****************************************************************
"""

def train_xgboost_model():

    df = pd.read_csv(os.getenv("CLEANED_SALES_DATA"))

    # cast categorical columns for native handling
    df["item_id"] = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    # define target and feature columns
    y = df["sales"]
    X = df[[
        "sell_price", "snap", "is_event_day", "lag_7", "rolling_mean_7",
        "day_of_week", "price_change_pct", "month", "item_id", "store_id"
    ]]

    # model hyperparameters
    model = xgboost.XGBRegressor(
        objective="reg:quantileerror",
        # 90th percentile target: 90% of values expected to be ≤ prediction
        quantile_alpha=0.9,
        tree_method="hist",
        enable_categorical=True,
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=30,
    )

    # how big the validation set should be from training data. We take 90% for training, 10% for validation.
    val_size = int(0.1 * len(X))

    """
    X_train: First 90% of feature data used to train the model
    y_train: First 90% of sales values corresponding to X_train

    X_val: Last 10% of feature data used as input to the model for prediction
    y_val: Last 10% of actual sales values, used to compare against predictions

    This setup simulates forecasting future sales using past data, 
    where we train on historical patterns and validate predictions 
    on unseen, more recent data — helping evaluate model performance realistically.
    """

    # training features ("sell_price", "snap", "is_event_day", "lag_7", "rolling_mean_7",
    #         "day_of_week", "price_change_pct", "month", "item_id", "store_id") (first 90%)
    X_train = X[:-val_size]
    # sales for training ("sales") (first 90%)
    y_train = y[:-val_size]

    # validation features ("sales") (last 10%)
    X_val = X[-val_size:]
    # actual sales for validation ("sales") (last 10%)
    y_val = y[-val_size:]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10,
    )

    # actually predict our last 10% sales based on the last 10% of features
    preds = model.predict(X_val)

    results = pd.DataFrame({
        "Actual": y_val.values,
        "Predicted": preds
    })

    print(results)
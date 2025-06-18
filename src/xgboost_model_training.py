import os
import pandas as pd
import xgboost

"""
    ****************************************************************
     Train and optimize the model including hyper-parameter tuning.
    ****************************************************************
"""


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
    quantile_alpha=0.9,             # 90th percentile target: 90% of values expected to be ≤ prediction
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
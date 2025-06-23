import pandas as pd
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

"""
****************************************************************
 Quantile Random Forest Model Training
 Note: Run standalone before randomforest_prediciton.py
****************************************************************
"""


# load cleaned input dataset
df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/sales_cleaned.csv")  # Replace this

# encode the categorical features
ordinal = OrdinalEncoder()
df[["item_id", "store_id"]] = ordinal.fit_transform(df[["item_id", "store_id"]])

# select all our features for training
FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id"
]
TARGET = "sales"

# assign features as x and target as y (sales)
X = df[FEATURES]
y = df[TARGET]

# first we want to split our training data and validation data, each with their own inputs/outputs
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# train quantile random forest regressor
qrf = RandomForestQuantileRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=10,
    random_state=42
)
qrf.fit(X_train, y_train)

# we want to predict 3 quantiles, 0.1, 0.5, 0.9
quantiles = [0.1, 0.5, 0.9]
preds = qrf.predict(X_val, quantiles=quantiles)
q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]

# MAE and RMSE evaluation of the QRF model
mae = mean_absolute_error(y_val, q50)
rmse = np.sqrt(mean_squared_error(y_val, q50))

print(f"Q50 (median) MAE:  {mae:.4f}")
print(f"Q50 (median) RMSE: {rmse:.4f}")

# save outputs
results = X_val.copy()
results["actual"] = y_val.values
results["q10"] = q10
results["q50"] = q50
results["q90"] = q90
results["in_band"] = (results["actual"] >= results["q10"]) & (results["actual"] <= results["q90"])

results.to_csv("quantile_rf_forecast_results.csv", index=False)
print("📁 Results saved to: quantile_rf_forecast_results.csv")
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


# --- Load your cleaned dataset ---
df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/final-project-implementation/data/processed/sales_cleaned.csv")  # Replace this

# --- Cast and encode categorical columns (ordinal encoding is RAM-safe) ---
ordinal = OrdinalEncoder()
df[["item_id", "store_id"]] = ordinal.fit_transform(df[["item_id", "store_id"]])

# --- Select features ---
FEATURES = [
    "sell_price", "is_event_day", "lag_7", "rolling_mean_7",
    "day_of_week", "month", "item_id", "store_id"
]
TARGET = "sales"

X = df[FEATURES]
y = df[TARGET]

# --- Optional: downsample for initial testing ---
# df = df.sample(500_000, random_state=42)

# --- Train/validation split ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Train Quantile Random Forest safely ---
qrf = RandomForestQuantileRegressor(
    n_estimators=100,       # Start small
    max_depth=10,           # Limit tree depth
    min_samples_leaf=10,    # Enforce minimum leaf size to reduce leaf count
    random_state=42
)
qrf.fit(X_train, y_train)

# --- Predict multiple quantiles ---
quantiles = [0.1, 0.5, 0.9]
preds = qrf.predict(X_val, quantiles=quantiles)
q10, q50, q90 = preds[:, 0], preds[:, 1], preds[:, 2]

# --- Evaluation ---
mae = mean_absolute_error(y_val, q50)
rmse = np.sqrt(mean_squared_error(y_val, q50))

print(f"Q50 (median) MAE:  {mae:.4f}")
print(f"Q50 (median) RMSE: {rmse:.4f}")

# --- Save results ---
results = X_val.copy()
results["actual"] = y_val.values
results["q10"] = q10
results["q50"] = q50
results["q90"] = q90
results["in_band"] = (results["actual"] >= results["q10"]) & (results["actual"] <= results["q90"])

results.to_csv("quantile_rf_forecast_results.csv", index=False)
print("📁 Results saved to: quantile_rf_forecast_results.csv")
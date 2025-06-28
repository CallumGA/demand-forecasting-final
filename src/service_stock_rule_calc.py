import pandas as pd
import math
from pathlib import Path

# *** Calculate the actual stock level we should use ***
def service_stock_level_calc(lower_bound: float,
                             upper_bound: float,
                             predicted_sales: float,
                             intv_confidence_zscore: float = 1.96,
                             stock_zscore: float = 1.28):
    # calculate estimated standard deviation for our upper and lower bands, using z-score of 95 % confidence
    standard_deviation = (upper_bound - lower_bound) / (2 * intv_confidence_zscore)

    # calculate target stock level, using forecasted sales, z-score of targeted stock-out prevention (90 %), standard deviation from above
    target_stock_level = predicted_sales + (stock_zscore * standard_deviation)

    return math.ceil(target_stock_level)
# -----------------------------------------------


def add_stock_levels_to_csv(csv_path: str | Path) -> None:
    csv_path = Path(csv_path).expanduser()
    df = pd.read_csv(csv_path)
    required_cols = {"predicted_sales", "lower_bound", "upper_bound"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    df["predicted_stock_level"] = df.apply(
        lambda r: service_stock_level_calc(
            r["lower_bound"], r["upper_bound"], r["predicted_sales"]
        ),
        axis=1,
    )

    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    add_stock_levels_to_csv(
        "/Users/callumanderson/Desktop/xgb_point_forecast_eval_predictions_full_20250625_181349.csv"
    )
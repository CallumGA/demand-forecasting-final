import os
from src import utils
import pandas as pd

"""
    ***********************************************
     Feature engineering
    ***********************************************
    Original Raw Features
        - (from cleaned sales data: sales, item_id, date, price, etc.)
    Engineered Features
        - is_event_day: Special event/holiday or no (from event_name_1/event_type_1/event_name_2/event_type_2)
        - lag_7: Captures weekly seasonality (from item_id, store_id, sales)
        - rolling_mean_7: Average weekly sales to smooth fluctuations (from item_id, store_id, sales)
        - day_of_week: Explains reoccurring behavior (from date)
"""


def apply_feature_engineering():
    df = utils.load_csv(os.getenv("CLEANED_SALES_DATA"))
    df = sort_by_date(df)
    df = add_single_event_feature(df)
    df = add_sales_lag(df)
    df = add_rolling_mean(df)
    df = add_day_of_week(df)
    df = remove_irrelevant_features(df)

    df.to_csv(os.getenv("CLEANED_SALES_DATA"), index=False)
    print(f"Saved updated feature set to: {os.getenv('CLEANED_SALES_DATA')}")


def sort_by_date(df):
    df = df.sort_values(by=["item_id", "store_id", "date"]).reset_index(drop=True)
    return df


def add_single_event_feature(df):
    df['is_event_day'] = df[['event_name_1', 'event_name_2']].notna().any(axis=1).astype(int)
    return df.drop(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis=1)


def add_sales_lag(df):
    df["lag_7"] = df.groupby(["item_id", "store_id"])["sales"].shift(7)
    return df


def add_rolling_mean(df):
    df["rolling_mean_7"] = (
        df.groupby(["item_id", "store_id"])["sales"]
          .shift(1)
          .rolling(window=7, min_periods=7)
          .mean()
    )
    return df


def add_day_of_week(df):
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    return df


def remove_irrelevant_features(df):
    # drop rolling/means/lags with no data ie; first 7 days
    df = df.dropna(subset=["lag_7", "rolling_mean_7"])
    # we already have date and d, which covers all of these. State and cat ID are noise.
    drop_cols = [
        "id", "dept_id", "cat_id", "state_id", "wm_yr_wk", "weekday", "wday", "year"
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    return df




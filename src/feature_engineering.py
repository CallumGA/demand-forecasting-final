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
        - snap: Is current state in a snap program day (from snap_CA, snap_TX, snap_WI)
        - is_event_day: Special event/holiday or no (from event_name_1/event_type_1/event_name_2/event_type_2)
        - lag_7: Captures weekly seasonality (from item_id, store_id, sales)
        - rolling_mean_7: Average weekly sales to smooth fluctuations (from item_id, store_id, sales)
        - day_of_week: Explains reoccurring behavior (from date)
        - price_change_pct: Calculates relative price change of each rows price compared to avg price for item across all time (from item_id, sell_price)
"""


def apply_feature_engineering():
    df = utils.load_csv(os.getenv("CLEANED_SALES_DATA"))
    df = sort_by_date(df)
    df = add_single_snap_feature(df)
    df = add_single_event_feature(df)
    df = add_sales_lag(df)
    df = add_rolling_mean(df)
    df = add_day_of_week(df)
    df = add_price_change_pct(df)
    df = remove_irrelevant_features(df)

    df.to_csv(os.getenv("CLEANED_SALES_DATA"), index=False)
    print(f"Saved updated feature set to: {os.getenv('CLEANED_SALES_DATA')}")


def sort_by_date(df):
    df = df.sort_values(by=["item_id", "store_id", "date"]).reset_index(drop=True)
    return df

def add_single_snap_feature(df):
    snap_map = {'CA': 'snap_CA', 'TX': 'snap_TX', 'WI': 'snap_WI'}
    df['snap'] = df.apply(lambda row: row[snap_map[row['state_id']]], axis=1)
    return df.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)


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


def add_price_change_pct(df):
    df["price_change_pct"] = df.groupby("item_id")["sell_price"].transform(lambda x: (x - x.mean()) / x.mean())
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


def is_weekend(df):
    ...
    # TODO: add is_weekend for better training outcome


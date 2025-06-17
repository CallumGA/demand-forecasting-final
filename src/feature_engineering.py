import os
from src import utils

"""
    ***********************************************
     Feature engineering
    ***********************************************
    Features:
    ------
    Original Raw Features
        - (from cleaned sales data: sales, item_id, date, price, etc.)
    Engineered Features
        - 'snap' from snap_CA, snap_TX, snap_WI
        - 'is_event_day' from event_name_1/event_type_1/event_name_2/event_type_2
        - More features coming: day_of_week, lag_7, rolling_mean_7, etc.
"""


def apply_feature_engineering():
    df = utils.load_csv(os.getenv("CLEANED_SALES_DATA"))
    df = add_single_snap_feature(df)
    df = add_single_event_feature(df)
    # df = add_sales_lag(df)
    # df = add_rolling_features(df)
    # df = add_date_features(df)
    df.to_csv(os.getenv("CLEANED_SALES_DATA"), index=False)
    print(f"Saved updated feature set to: {os.getenv('CLEANED_SALES_DATA')}")


def add_single_snap_feature(df):
    snap_map = {'CA': 'snap_CA', 'TX': 'snap_TX', 'WI': 'snap_WI'}
    df['snap'] = df.apply(lambda row: row[snap_map[row['state_id']]], axis=1)
    return df.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1)


def add_single_event_feature(df):
    df['is_event_day'] = df[['event_name_1', 'event_name_2']].notna().any(axis=1).astype(int)
    return df.drop(['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'], axis=1)


def add_sales_lag(df):
    # Placeholder for lag feature engineering
    return df


def add_rolling_features(df):
    # Placeholder for rolling average feature engineering
    return df


def add_date_features(df):
    # Placeholder for date-based features like is_weekend, day_of_week, etc.
    return df


def remove_irrelevant_features(df):
    # Remove irrelevant features that will just be noise
    return df


def build_training_matrix():
    """
    Prepare final dataset for model input by encoding categorical features.
    This function can be expanded based on modeling needs.
    """
    pass  # To be implemented
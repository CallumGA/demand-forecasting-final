import os
import pandas as pd
from src import utils

"""
    ***********************************************
     Feature engineering
    ***********************************************
    Features:
    ------
    Original Raw Features
        -
        -
    Engineered Features
        - snap_CA, snap_TX, and snap_WI features into a single 'snap' feature
        -
"""
# TODO: next we want to engineer features out of the other csvs into the main sales_cleaned.csv: price, day_of_week, special_event_1, special_event_2, lag_7, rolling_mean_7 and parse to sales_cleaned.parquet


def main():
    snap_df = utils.load_csv(os.getenv("CLEANED_SALES_DATA"))
    merge_snap_feature(snap_df)


def merge_snap_feature(df):
    snap_map = {'CA': 'snap_CA', 'TX': 'snap_TX', 'WI': 'snap_WI'}
    df['snap'] = df.apply(lambda row: row[snap_map[row['state_id']]], axis=1)
    df.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis=1, inplace=True)

    df.to_csv(os.getenv("CLEANED_SALES_DATA"), index=False)


import os
from src import utils

"""
**************************************************
 Data Preprocessing: Load, clean, and prepare raw data
**************************************************
"""


def run_data_cleaning_pipeline():
    extract_raw_data()
    sanitize_calendar_data()
    sanitize_sales_data()
    sanitize_prices_data()
    melt_sales_data()


def extract_raw_data():
    utils.unzip_file(os.getenv("RAW_DATA_PATH"))


def sanitize_calendar_data():
    df = utils.load_csv(os.getenv("CALENDAR_DATA_PATH"))
    utils.validate_dataframe_integrity(df)
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        df[col] = df[col].fillna("None")
    df.to_csv(os.getenv("CALENDAR_DATA_PATH"), index=False)
    print(f"Calendar data cleaned and saved: {os.getenv('CALENDAR_DATA_PATH')}")


def sanitize_prices_data():
    df = utils.load_csv(os.getenv("SELL_PRICES_DATA_PATH"))
    utils.validate_dataframe_integrity(df)
    print(f"Prices data verified: {os.getenv('SELL_PRICES_DATA_PATH')}")


def sanitize_sales_data():
    df = utils.load_csv(os.getenv("SALES_VALIDATION_DATA_PATH"))
    utils.validate_dataframe_integrity(df)
    print(f"Sales data verified: {os.getenv('SALES_VALIDATION_DATA_PATH')}")


def melt_sales_data():
    sales_df = utils.load_csv(os.getenv("SALES_VALIDATION_DATA_PATH"))
    calendar_df = utils.load_csv(os.getenv("CALENDAR_DATA_PATH"))
    prices_df = utils.load_csv(os.getenv("SELL_PRICES_DATA_PATH"))
    melted_df = sales_df.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d',
        value_name='sales'
    )
    melted_df = melted_df.merge(calendar_df, how='left', on='d')
    melted_df = melted_df.merge(prices_df, how='left', on=['store_id', 'item_id', 'wm_yr_wk'])
    melted_df = melted_df.sort_values(by=['id', 'd']).reset_index(drop=True)
    melted_df.to_csv(os.getenv("CLEANED_SALES_DATA"), index=False)
    print(f"Melted and saved cleaned data to: {os.getenv('CLEANED_SALES_DATA')}")
    print(f"Days covered: {melted_df['d'].nunique()} | From {melted_df['date'].min()} to {melted_df['date'].max()}")

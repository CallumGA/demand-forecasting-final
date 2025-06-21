import os
import pandas as pd
from src import utils

"""
*************************************************************
M5  →  cleaned_sales.csv
  • validates & cleans calendar / prices / sales raw files
  • melts sales to long format
  • engineers all trainer features
  • writes CLEANED_SALES_DATA ready for XGBoost trainer
*************************************************************
"""


RAW_ZIP_PATH        = os.getenv("RAW_DATA_PATH",               "./data/raw/data.zip")
CALENDAR_PATH       = os.getenv("CALENDAR_DATA_PATH",          "./data/raw/calendar.csv")
PRICES_PATH         = os.getenv("SELL_PRICES_DATA_PATH",       "./data/raw/sell_prices.csv")
SALES_VAL_WIDE_PATH = os.getenv("SALES_VALIDATION_DATA_PATH",  "./data/raw/sales_train_validation.csv")
SALES_EVAL_WIDE_PATH= os.getenv("SALES_EVALUATION_DATA_PATH",  "./data/raw/sales_train_evaluation.csv")

CLEANED_OUT_PATH    = os.getenv("CLEANED_SALES_DATA",          "./data/processed/sales_cleaned.csv")
OUT_PATH            = CLEANED_OUT_PATH

ENCODER_DIR         = os.getenv("ENCODER_DIR", "./models/encoders")
SAVED_MODELS_DIR    = os.getenv("SAVED_MODELS", "./models/")
LOG_DIR             = os.getenv("LOG_DIR", "./logs")


FEATURES = [
    "item_id", "store_id", "d", "sales", "date",
    "month", "day_of_week",
    "snap_CA", "snap_TX", "snap_WI",
    "sell_price", "is_event_day",
    "lag_7", "rolling_mean_7",
]


def _validate_final(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for f in FEATURES:
        if df[f].isna().mean() > 0.10:
            print(f"Warning: {f} has {df[f].isna().mean():.1%} NaN values.")
    df = df.dropna(subset=FEATURES)

    df["item_id"]  = df["item_id"].astype("category")
    df["store_id"] = df["store_id"].astype("category")

    df = df.sort_values(["item_id", "store_id", "date"]).reset_index(drop=True)

    return df[FEATURES]


def extract_raw_data():
    if RAW_ZIP_PATH and os.path.exists(RAW_ZIP_PATH):
        utils.unzip_file(RAW_ZIP_PATH)

def clean_calendar():
    df = utils.load_csv(CALENDAR_PATH)
    utils.validate_dataframe_integrity(df)
    for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
        df[col] = df[col].fillna("None")
    df.to_csv(CALENDAR_PATH, index=False)
    print("✓ calendar cleaned")

def clean_prices():
    df = utils.load_csv(PRICES_PATH)
    utils.validate_dataframe_integrity(df)
    print("✓ prices verified")

def clean_sales_wide():
    df = utils.load_csv(SALES_VAL_WIDE_PATH)
    utils.validate_dataframe_integrity(df)
    print("✓ sales (wide) verified")


def melt_and_engineer():
    sales_wide = utils.load_csv(SALES_VAL_WIDE_PATH)
    cal        = utils.load_csv(CALENDAR_PATH)
    prices     = utils.load_csv(PRICES_PATH)

    sales_long = sales_wide.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id',
                 'store_id', 'state_id'],
        var_name='d',
        value_name='sales'
    )

    df = sales_long \
        .merge(cal,    on='d', how='left') \
        .merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left') \
        .sort_values(['id', 'd']) \
        .reset_index(drop=True)

    df['date']         = pd.to_datetime(df['date'])
    df['month']        = df['date'].dt.month.astype("int8")
    df['day_of_week']  = df['date'].dt.dayofweek.astype("int8")
    df['is_event_day'] = ((df['event_name_1'] != "None") |
                          (df['event_name_2'] != "None")).astype("int8")

    df['lag_7'] = df.groupby(['item_id', 'store_id'], observed=True)['sales'] \
                    .shift(7)
    df['rolling_mean_7'] = df.groupby(['item_id', 'store_id'], observed=True)['sales'] \
                              .shift(1) \
                              .rolling(window=7, min_periods=7) \
                              .mean()

    df_clean = _validate_final(df)
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)

    print(f"✓ cleaned_sales.csv written → {OUT_PATH}")
    print(f"Rows: {len(df_clean):,} | Days: {df_clean['d'].nunique()} "
          f"| {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")


def run_data_cleaning_pipeline():
    extract_raw_data()
    clean_calendar()
    clean_prices()
    clean_sales_wide()
    melt_and_engineer()


if __name__ == "__main__":
    run_data_cleaning_pipeline()
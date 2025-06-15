import os
import joblib
from src import utils
from sklearn.preprocessing import LabelEncoder

"""
    ****************************************************************************
     Clean, sanitize, and encode data for categorical features before training.
    ****************************************************************************
"""


def extract_raw_data():
    utils.unzip_file(os.getenv("RAW_DATA_PATH"))


def sanitize_raw_calendar_data():
    calendar_data = utils.load_csv(os.getenv("CALENDAR_DATA_PATH"))
    utils.validate_dataframe_integrity(calendar_data)
    event_columns = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    encoder_dir = os.getenv("ENCODER_DIR", "models/encoders")
    os.makedirs(encoder_dir, exist_ok=True)
    for col in event_columns:
        calendar_data[col] = calendar_data[col].fillna('None')
        le = LabelEncoder()
        calendar_data[col] = le.fit_transform(calendar_data[col])
        encoder_path = os.path.join(encoder_dir, f"{col}_encoder.pkl")
        joblib.dump(le, encoder_path)
        print(f"Saved encoder: {encoder_path}")
    calendar_data.to_csv(os.getenv('CALENDAR_DATA_PATH'), index=False)
    print(f"Cleaned and saved to: {os.getenv('CALENDAR_DATA_PATH')}")


def sanitize_raw_prices_data():
    sales_data = utils.load_csv(os.getenv("SELL_PRICES_DATA_PATH"))
    utils.validate_dataframe_integrity(sales_data)


def sanitize_raw_sales_data():
    prices_data = utils.load_csv(os.getenv("SALES_VALIDATION_DATA_PATH"))
    utils.validate_dataframe_integrity(prices_data)


def melt_data():
    sales_df = utils.load_csv(os.getenv("SALES_VALIDATION_DATA_PATH"))
    calendar_df = utils.load_csv(os.getenv("CALENDAR_DATA_PATH"))
    prices_df = utils.load_csv(os.getenv("SELL_PRICES_DATA_PATH"))
    melted_sales = sales_df.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d',
        value_name='sales'
    )
    melted_sales = melted_sales.merge(calendar_df, how='left', on='d')
    melted_sales = melted_sales.merge(prices_df, how='left',
                                      on=['store_id', 'item_id', 'wm_yr_wk'])
    melted_sales = melted_sales.sort_values(by=['id', 'd']).reset_index(drop=True)
    melted_sales.to_csv(os.getenv("CLEANED_SALES_DATA"), index=False)
    print(f"Melted and saved cleaned data file to: {os.getenv('CLEANED_SALES_DATA')}. Ready for converting to training matrix.")
    print(melted_sales['d'].nunique())
    print(melted_sales['date'].min(), melted_sales['date'].max())


def build_training_matrix():
    """
    Prepare CSV into training matrix by encoding non-numerical columns.
    """
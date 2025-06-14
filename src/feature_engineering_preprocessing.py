import os
from src import utils
from sklearn.preprocessing import LabelEncoder

"""
    ***********************************************
     Feature engineering and data preprocessing code.
    ***********************************************
    Features:
    ------
    Original Raw Features
        - 
        -
    Engineered Features
        -
        -     
"""

def extract_raw_data():
    utils.unzip_file(os.getenv("RAW_DATA_PATH"))


def sanitize_raw_calendar_data():
    calendar_data = utils.load_csv(os.getenv("CALENDAR_DATA_PATH"))
    utils.validate_dataframe_integrity(calendar_data)
    event_columns = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_columns:
        calendar_data[col] = calendar_data[col].fillna('None')
        le = LabelEncoder()
        calendar_data[col] = le.fit_transform(calendar_data[col])
    calendar_data.to_csv(os.getenv('CALENDAR_DATA_PATH'), index=False)
    print(f"Cleaned and saved to: {os.getenv('CALENDAR_DATA_PATH')}")


def sanitize_raw_prices_data():
    sales_data = utils.load_csv(os.getenv("SELL_PRICES_DATA_PATH"))
    utils.validate_dataframe_integrity(sales_data)


def sanitize_raw_sales_data():
    prices_data = utils.load_csv(os.getenv("SALES_VALIDATION_DATA_PATH"))
    utils.validate_dataframe_integrity(prices_data)



# TODO: then we write to the cleaned final csv in data/processed/sales_cleaned.parquet

# TODO: next we want to rip features out of the other csvs into the main sales_cleaned.parquet: price, day_of_week, is_holiday, lag_7, rolling_mean_7

# TODO: melt to from wide format to long format and finally we want to append the new feature rows to data/processed/sales_cleaned.parquet

# TODO: we also must melt sales_train_evaluation.csv to look exactly like the preprocessed final csv which we train the model with

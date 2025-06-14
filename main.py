from dotenv import load_dotenv

from src import feature_engineering_preprocessing

"""
    ***********************************************
     Main orchestrator for the project
    ***********************************************
"""

load_dotenv()

# extract raw data
feature_engineering_preprocessing.extract_raw_data()

# sanitize the raw data (remove empty/NaN/malformed and encode holiday/promo in calendar.csv)
feature_engineering_preprocessing.sanitize_raw_calendar_data()
feature_engineering_preprocessing.sanitize_raw_prices_data()
feature_engineering_preprocessing.sanitize_raw_sales_data()
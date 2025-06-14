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

# sanitize the raw data (remove empty/NaN/malformed)
feature_engineering_preprocessing.sanitize_raw_data()
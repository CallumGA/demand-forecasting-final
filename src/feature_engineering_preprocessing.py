import pandas as pd

"""
    ***********************************************
     Feature engineering and data preprocessing code.
    ***********************************************
    Features:
    ------
    Original Raw Features
        If the zip file does not exist.
    Engineered Features
        If the file is not a valid zip archive.        
    
"""

# TODO: unzip the raw data in data/raw/data.zip

# TODO: we need to first sanitize the csv by making sure there are no empty rows/columns/cells

# TODO: then we write to the cleaned final csv in data/processed/sales_cleaned.parquet

# TODO: next we want to rip features out of the other csvs into the main sales_cleaned.parquet: price, day_of_week, is_holiday, lag_7, rolling_mean_7

# TODO: finally we want to append the new feature rows to data/processed/sales_cleaned.parquet

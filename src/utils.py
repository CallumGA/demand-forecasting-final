import pandas as pd

"""
    ***********************************************
     Utility functions for use in the main project 
    ***********************************************
"""

def load_selected_columns(csv_path: str, columns: list) -> pd.DataFrame:
    """
    Loads specified columns from a CSV file into a pandas DataFrame.
    """

    full_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    missing = [col for col in columns if col not in full_columns]
    if missing:
        raise ValueError(f"The following columns were not found in the CSV: {missing}")

    return pd.read_csv(csv_path, usecols=columns)
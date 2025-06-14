import os
import zipfile

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


def unzip_file(zip_path: str, extract_to: str = None) -> None:
    """
    Extracts the contents of a ZIP file.
    """

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if extract_to is None:
        extract_to = os.path.dirname(zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted to: {extract_to}")
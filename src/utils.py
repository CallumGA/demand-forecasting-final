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


def load_csv(csv_path: str, **kwargs) -> pd.DataFrame:
    """
    Loads an entire CSV file into a pandas DataFrame.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, **kwargs)
        print(f"Loaded {csv_path} with shape {df.shape}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading CSV file: {e}")


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


def validate_dataframe_integrity(df: pd.DataFrame, raise_on_error: bool = False) -> bool:
    """
    Validates that the DataFrame has no missing (NaN) values or empty string cells.
    """

    issues_found = False

    if df.isnull().values.any():
        issues_found = True
        msg = f"Data contains missing (NaN) values."
        if raise_on_error:
            raise ValueError(msg)
        else:
            print(msg)

    empty_str_mask = df.astype(str).apply(lambda x: x.str.strip() == '')
    if empty_str_mask.any().any():
        issues_found = True
        msg = f"Data contains empty string cells in columns."
        if raise_on_error:
            raise ValueError(msg)
        else:
            print(msg)

    if not issues_found:
        print("DataFrame passed integrity check: no NaNs or empty string cells.")
    return not issues_found
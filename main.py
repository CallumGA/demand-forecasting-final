from dotenv import load_dotenv

from src import feature_engineering_preprocessing

"""
    ***********************************************
     Main orchestrator for the project
    ***********************************************
"""

load_dotenv()

feature_engineering_preprocessing.extract_raw_data()
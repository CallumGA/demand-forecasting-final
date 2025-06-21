from dotenv import load_dotenv
from src import data_preprocess_validation, xgboost_model_training

"""
**************************************************
 Entry point for full ML pipeline
**************************************************
"""

def main():
    print("Starting ML pipeline...\n")

    # Load environment variables
    load_dotenv()

    # Step 1: Data extraction and cleaning
    print("Extracting and cleaning raw data...\n")
    #data_preprocess.run_data_cleaning_pipeline()

    # Step 2: Model training
    print("Training model...\n")
    xgboost_model_training.train_point_forecast()

    print("\nML pipeline completed successfully.\n")

if __name__ == "__main__":
    main()
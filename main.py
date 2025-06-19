import joblib
from dotenv import load_dotenv
from src import data_preprocess, feature_engineering, xgboost_model_training

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
    # data_preprocess.run_data_cleaning_pipeline()

    # Step 2: Feature engineering
    print("Engineering features...\n")
    # feature_engineering.apply_feature_engineering()

    # Step 3: Model training
    print("Training model...\n")
    xgboost_model_training.train_all_quantiles()

    # Step 4: Forecasting
    print("Forecasting next 28 days...\n")
    # forecast.run_forecast(model)

    print("\nML pipeline completed successfully.\n")

if __name__ == "__main__":
    main()
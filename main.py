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
    print("Extracting and cleaning raw data...")
    # data_preprocess.run_data_cleaning_pipeline()

    # Step 2: Feature engineering
    print("Engineering features...")
    # feature_engineering.apply_feature_engineering()

    # Step 3: Build training matrix
    print("Generating training feature matrix...")
    # data_preprocess.build_training_matrix()

    # Step 4: Model training
    print("Training model...")
    model = xgboost_model_training.train_xgboost_model()

    # Step 5: Forecasting
    print("Forecasting next 28 days...")
    # forecast.run_forecast(model)

    print("\nML pipeline completed successfully.")

if __name__ == "__main__":
    main()
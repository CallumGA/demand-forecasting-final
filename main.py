import joblib
from dotenv import load_dotenv
from src import data_preprocess, feature_engineering

"""
    ***********************************************
     Entry point for full ML pipeline
    ***********************************************
"""


def main():
    print("Starting ML pipeline...\n")

    # Load environment variables
    load_dotenv()

    # Extract and sanitize raw data
    print("Extracting and cleaning raw data...")
    data_preprocess.extract_raw_data()
    data_preprocess.sanitize_raw_calendar_data()
    data_preprocess.sanitize_raw_sales_data()
    data_preprocess.sanitize_raw_prices_data()
    data_preprocess.melt_data()

    # Feature engineering (building and appending final data file with custom engineered features)
    # feature_engineering.parse_raw_features()

    # Build training feature matrix and save to /data/processed
    print("Generating training feature matrix...")
    # data_preprocess.build_training_feature_matrix()

    # Train model on cleaned feature set
    print("Training model...")
    # model = model_training.train_model()

    # Run inference on evaluation window
    print("Forecasting next 28 days...")
    # forecast.run_forecast(model)

    print("\nML pipeline completed successfully.")

if __name__ == "__main__":
    main()
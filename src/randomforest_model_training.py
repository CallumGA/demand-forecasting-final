import os
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator

"""
****************************************************************
 Quantile Random Forest Model Training (Stratified + Fast)
****************************************************************
"""

# TODO: must be trained on the same data split as xgboost model (groupwise-time split)
# TODO: comment and explain each function
# TODO: compute baseline
# TODO: potentially add interval predictions


def train_and_evaluate_h2o_rf(csv_file, target_col='sales', validation_split=0.2):
    h2o.init(max_mem_size="16G", nthreads=-1)

    print("Loading data...")
    df = h2o.import_file(csv_file)
    print(f"Dataset shape: {df.shape}")

    categorical_cols = ['item_id', 'store_id']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].asfactor()

    train, test = df.split_frame(ratios=[1 - validation_split], seed=42)
    feature_cols = [col for col in df.columns if col != target_col]

    print("Training Random Forest...")
    rf = H2ORandomForestEstimator(
        ntrees=100,
        max_depth=20,
        min_rows=10,
        seed=42
    )
    rf.train(x=feature_cols, y=target_col, training_frame=train)

    print("Making predictions...")
    predictions = rf.predict(test)

    test_rmse = rf.rmse(valid=False, train=False)
    test_mae = rf.mae(valid=False, train=False)

    print(f"\nResults:\nRMSE: {test_rmse:.4f}\nMAE: {test_mae:.4f}")

    h2o.shutdown(prompt=False)
    return test_rmse, test_mae


if __name__ == "__main__":
    file_path = os.path.expanduser("~/h2o_data/sales_cleaned.csv")
    rmse, mae = train_and_evaluate_h2o_rf(file_path)
    print(f"Final - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
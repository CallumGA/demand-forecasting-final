import pandas as pd
from scipy.ndimage import standard_deviation
import math

# read model outputted csv for our results columns
df = pd.read_csv("/Users/callumanderson/Documents/Documents - Callum’s Laptop/Masters-File-Repo/MIA5130/final-project/milestone2/xgboost_evaluations/xgb_point_forecast_eval_predictions_full_20250625_181349.csv",
                 usecols=["predicted_sales", "lower_bound", "upper_bound"])


# service stock level formula for taking our prediction, upper & lower bounds and converting to a usable stock prediction
def service_stock_level_calc(lower_bound: float, upper_bound: float, predicted_sales: float, intv_confidence_zscore = 1.96, stock_zscore = 1.75):

    # calculate estimated standard deviation for our upper and lower bands, using z-score of 95% confidence
    standard_deviation = (upper_bound - lower_bound) / (2 * intv_confidence_zscore)

    # calculate target stock level, using forecasted sales, z-score of targeted stockout prevention (96%), standard deviation from above
    target_stock_level = predicted_sales + (stock_zscore * standard_deviation)

    return math.ceil(target_stock_level)




# *** Entry point ***
def calc():
    return service_stock_level_calc(0.0, 11.728143692016602, 4.109958)


if __name__ == "__main__":
    print(f"Suggested stock level is: {calc()}")
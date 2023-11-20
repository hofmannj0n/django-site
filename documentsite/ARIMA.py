import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

class ARIMAModel:
    def __init__(self, p, d, q):
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def fit(self, data):
        self.model = SARIMAX(data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()

    def forecast(self, steps):
        return self.model_fit.forecast(steps)

    @staticmethod
    def difference(data, interval=1):
        return data.diff(interval).dropna(inplace=False)

    @staticmethod
    def reverse_difference(history, yhat, interval=1):
        return np.r_[history.iloc[0], yhat].cumsum()

    @staticmethod
    def evaluate_forecast(test, predictions):
        mse = mean_squared_error(test, predictions)
        return mse

    @staticmethod
    def evaluate_smape(test, predictions):
        absolute_percentage_error = np.abs(predictions - test) * 200 / (np.abs(predictions) + np.abs(test))
        mean_smape = np.mean(absolute_percentage_error)
        return mean_smape
    
    @staticmethod
    def evaluate_forecast(test_series, reverse_test_diff, reverse_predictions):
        if len(test_series) == len(reverse_test_diff) and len(test_series) == len(reverse_predictions):
            mse = mean_squared_error(reverse_test_diff, reverse_predictions)
            return mse
        else:
            raise ValueError("Input variables have inconsistent numbers of samples.")

    @staticmethod
    def plot_forecast(test_series, reverse_test_diff, reverse_predictions):
        plt.figure(figsize=(12, 7))
        plt.title('Prices (Test / Forecast)')
        plt.xlabel('Dates')
        plt.ylabel('Prices')
        plt.plot(reverse_test_diff, color='green', marker='.', label='Testing Prices - Reverse Diff Transform')
        plt.plot(reverse_predictions, color='red', linestyle='--', label='Forecasted Prices - Reverse Diff Transform')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Example usage
    p, d, q = 5, 1, 0  # Adjust these values based on your requirements
    arima_model = ARIMAModel(p, d, q)

    symbol = input("Enter in Ticker Symbol: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")

    stock_data = yf.download(symbol, start=start_date, end=end_date)

    df = pd.DataFrame(stock_data)

    df = df.drop('Adj Close', axis=1)
    df['symbol'] = symbol
    df.isnull().values.any()
    df = df.dropna()

    train_data, test_data = df.iloc[0:int(len(df) * 0.8), :], df.iloc[int(len(df) * 0.8):, :]
    train_series = train_data['Open']
    test_series = test_data['Open']

    # Example usage of ARIMAModel
    train_diff = ARIMAModel.difference(train_series)
    arima_model.fit(train_diff)

    # Assuming you have the test series
    test_series = test_data['Open']

    # Difference the test series
    test_diff = ARIMAModel.difference(test_series)

    # Forecast using the ARIMA model
    steps = len(test_data)
    predictions = arima_model.forecast(steps)

    # Ensure predictions are aligned with the length of the original test series
    predictions = predictions[:len(test_series)]

    # Reverse the differences
    reverse_test_diff = ARIMAModel.reverse_difference(train_series, test_diff)
    reverse_predictions = ARIMAModel.reverse_difference(train_series, predictions)

    # Calculate metrics
    mse = ARIMAModel.evaluate_forecast(test_series, reverse_predictions)
    smape = ARIMAModel.evaluate_smape(test_series, reverse_predictions)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"SMAPE: {smape}")

    ARIMAModel.plot_forecast(test_series, reverse_test_diff, reverse_predictions)

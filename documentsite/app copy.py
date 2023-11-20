import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# process the data

symbol = input("Enter in Ticker Symbol: ")
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

data = yf.download(symbol, start=start_date, end=end_date)

df = pd.DataFrame(data)

df = df.drop('Adj Close', axis=1)
df['symbol'] = symbol
df.isnull().values.any()
df = df.dropna()

train_data, test_data = df.iloc[0:int(len(df)*0.8), :], df.iloc[int(len(df)*0.8):, :]

window = 7
train_series = train_data['Open']

#Determing rolling statistics
rolmean = train_series.rolling(window).mean()
rolstd = train_series.rolling(window).std()

dftest = adfuller(train_series, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
dfoutput

# Applying first-order differencing
train_diff = train_series.diff(periods=1)
train_diff = train_diff.dropna(inplace = False)

#Determing rolling statistics
window = 7

rolling_mean = train_diff.rolling(window).mean()
rolling_std = train_diff.rolling(window).std()

dftest = adfuller(train_diff, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
dfoutput

def smape_kun(y_true, y_pred):
    absolute_percentage_error = np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))
    mean_smape = np.mean(absolute_percentage_error)
    return mean_smape

test_series = test_data['Open']
test_diff = test_series.diff(periods=1)
test_diff = test_diff.dropna(inplace = False)

# Initialize history with training data
history = [x for x in train_diff]
predictions = list()

# Iterate through the test data points
for t in range(len(test_diff)):
    # START_CODE_HERE
    p, d, q = 5,1,0
    model = ARIMA(history, order=(p, d, q))
    model_fit = model.fit()
    # END_CODE_HERE

    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)

    obs = test_diff[t]
    history.append(obs)

    if t % 100 == 0:
        print('Test Series Point: {}\tPredicted={}, Expected={}'.format(t, yhat, obs))

# Calculate Mean Squared Error (MSE) to evaluate model performance
mse = mean_squared_error(test_diff, predictions)

reverse_test_diff = np.r_[test_series.iloc[0], test_diff].cumsum()
reverse_predictions = np.r_[test_series.iloc[0], predictions].cumsum()
reverse_test_diff.shape, reverse_predictions.shape

error = mean_squared_error(reverse_test_diff, reverse_predictions)
error2 = smape_kun(reverse_test_diff, reverse_predictions)

reverse_test_diff_series = pd.Series(reverse_test_diff)
reverse_test_diff_series.index = test_series.index

reverse_predictions_series = pd.Series(reverse_test_diff)
reverse_predictions_series.index = test_series.index

plt.figure(figsize=(12,7))
plt.title('IBM Prices (Test / Forecast)')
plt.xlabel('Dates')
plt.ylabel('Prices')
plt.plot(reverse_test_diff_series, color='green', marker='.', label='Testing Prices - Reverse Diff Transform')
plt.plot(reverse_test_diff_series, color='red', linestyle='--', label='Forecasted Prices - Reverse Diff Transform')
plt.legend();
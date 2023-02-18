import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load time series data into a Pandas DataFrame
data = pd.read_csv('time_series_data.csv', index_col='Date', parse_dates=True)

# Split the data into training and testing sets
train_data = data.iloc[:-12]
test_data = data.iloc[-12:]

# Create an ARIMA model with (p,d,q) parameters
model = ARIMA(train_data, order=(2,1,2))

# Fit the ARIMA model to the training data
model_fit = model.fit()

# Generate one-step ahead forecast for the test data
forecast = model_fit.forecast(steps=len(test_data))

# Convert the forecasted values into a Pandas Series with date index
forecast_series = pd.Series(forecast, index=test_data.index)

# Evaluate the accuracy of the forecast
accuracy = (forecast_series - test_data) / test_data * 100

# Print the forecast and accuracy results
print('Forecasted values:\n', forecast_series)
print('Accuracy:\n', accuracy)

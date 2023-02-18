import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Load time series data into a DataFrame in Pandas
data = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

### check for stationarity using Dickey-Fuller test ### 

plt.plot(data)
plt.title('time series distribution on sales data')
plt.xlabel('time')
plt.ylabel('value')
plt.show()

result = adfuller(data['Value'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])

for key, value in result[4].items():
    print('Critical Value at %s: %.3f' % (key, value))

### The test returns a test statistic and a p-value, and we can compare the test statistic to critical values at different significance levels to determine whether the null hypothesis can be rejected or not. 
### If the p-value is less than a chosen significance level (such as 0.05), we can reject the null hypothesis and conclude that the data is stationary.


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

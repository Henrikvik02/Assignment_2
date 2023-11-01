import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

file_path = 'TSLA.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data.asfreq('D', method='ffill')

train = data.iloc[:-30]  # All data except the last 30 days
test = data.iloc[-30:]   # Only the last 30 days

p, d, q = 1, 1, 1          # Non-seasonal parameters
P, D, Q, S = 1, 1, 1, 7   # Seasonal parameters

model = SARIMAX(train['Close'],
                order=(p, d, q),
                seasonal_order=(P, D, Q, S),
                enforce_stationarity=False,
                enforce_invertibility=False)

results = model.fit(disp=-1)

last_date_in_dataset = data.index[-1]
end_of_november_30 = pd.Timestamp('2023-11-30')
days_to_forecast = (end_of_november_30 - last_date_in_dataset).days + 30


print(f"Last date in dataset: {last_date_in_dataset}")
print(f"End of November 30: {end_of_november_30}")
print(f"Days to forecast: {days_to_forecast}")

extended_forecast = results.get_forecast(steps=days_to_forecast)
extended_forecast_mean = extended_forecast.predicted_mean

plt.figure(figsize=(15, 6))
plt.plot(data['Close'], label='Data')
plt.plot(extended_forecast_mean.index, extended_forecast_mean, label='Extended SARIMAX Forecast', color='red')
plt.axvline(x=end_of_november_30, color='gray', linestyle='--', label='30th November')
plt.legend(loc='best')
plt.title('Extended SARIMAX Forecast up to 30th November')
plt.show()

if end_of_november_30 in extended_forecast_mean.index:
    print(f"Predicted closing price for TSLA on 30th November 2023: ${extended_forecast_mean[end_of_november_30]:.2f}")
else:
    print(f"No prediction available for 30th November 2023.")

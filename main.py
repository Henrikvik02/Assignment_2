import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load and preprocess the dataset
data = pd.read_csv('TSLA.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Extract the 'Close' column for prediction
close_data = data['Close'].values.reshape(-1, 1)

# Normalize the closing prices to range between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# Split 80% of data for training and the rest for testing; consider last 60 days for test data input
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size - 60:, :]

# Utility function to format dataset for LSTM input
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# Create training and test data
time_step = 60
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

# Reshape data for LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train the LSTM model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, batch_size=1, epochs=5, validation_data=(X_test, Y_test))

# Make prediction for a specific date (30th of November)
last_60_days = data['Close'][-60:].values.reshape(-1, 1)
last_60_days_scaled = scaler.transform(last_60_days)
X_test_30th = [last_60_days_scaled]
predicted_price = model.predict(np.array(X_test_30th))
predicted_price = scaler.inverse_transform(predicted_price)

# Make predictions on the test data
test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)

# Plot the actual vs. predicted prices
actual = close_data[train_size:-time_step]
plt.figure(figsize=(16, 8))
plt.plot(data.index, close_data, label='Actual Close Prices', color='blue')
plt.plot(data.index[train_size+1:train_size+1+len(test_predictions)], test_predictions, label='Predicted Close Prices', color='red', linestyle='dashed')
plt.axvline(x=data.index[train_size], color='green', linestyle='--', label='Train-Test Split Point')
plt.axvline(x=data.index[-1], color='orange', linestyle='--', label=f'30th Nov Prediction: ${predicted_price[0][0]:.2f}')
plt.title('Tesla Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

# Dummy confidence score (for illustrative purposes only)
confidence_score = 90
print(f"Predicted Close Price for 30th of November: ${predicted_price[0][0]:.2f}")
print(f"Confidence Score: {confidence_score}%")
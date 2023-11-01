import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'TSLA.csv'
data = pd.read_csv(file_path)

# Convert date to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date
data = data.sort_values(by='Date')

# Convert date to numerical format
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

# Split the data into input features (X) and target variable (y)
X = data[['Date']]
y = data['Close']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean squared error:', mse)
print('R-squared value:', r2)


def predict_stock_price(date):
    # Convert date to datetime format
    date = pd.to_datetime(date)

    # Convert date to numerical format
    date_ordinal = pd.Timestamp.toordinal(date)

    # Make prediction
    predicted_price = model.predict([[date_ordinal]])[0]

    # Calculate prediction percentage score
    score = model.score(X_test, y_test) * 100

    return predicted_price, score


# Test the function
date = '2023-11-30'
predicted_price, score = predict_stock_price(date)
print('Predicted closing price on {}: ${:.2f}'.format(date, predicted_price))
print('Prediction percentage score: {:.2f}%'.format(score))


# Make predictions on the entire dataset
y_pred_all = model.predict(X)

# Convert ordinal dates back to datetime format for plotting
dates = [datetime.fromordinal(int(date)) for date in data['Date']]

# Plot the actual and predicted closing prices with formatted x-axis ticks
plt.figure(figsize=(10, 5))
plt.plot(dates, data['Close'], label='Actual')
plt.plot(dates, y_pred_all, label='Predicted', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Actual vs Predicted Closing Prices of Tesla Stock')
plt.legend()
plt.grid(True)

# Format x-axis ticks
plt.xticks(rotation=45)
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))

plt.tight_layout()
plt.show()
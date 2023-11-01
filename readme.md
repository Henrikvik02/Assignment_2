Assignement 2: Machine learning

In the assignement we was asked to only create one machine learning algorithm.
But we tought it fun to test if our group could create more, so we did. 
These are the following algorithms:
- Regression 
- Long Short-Term Memory (LSTM)
- SERIMAX 
- ChatGPT

In the main.py is the Long Short-Term Memory (LSTM)
- This model utilizes Long Short-Term Memory (LSTM), 
    a recurrent neural network architecture, 
    to predict Tesla's stock closing prices based on historical data.
- The provided data is first loaded and preprocessed, 
    with dates converted into a usable format for the model.
- The model specifically targets the 'Close' price of Tesla stocks for its predictions.
- Data normalization is performed to ensure the input values are in a consistent range, 
    aiding model training and performance.
- The dataset is divided into training and test sets, 
    with 80% used for training and the rest for testing.
- An LSTM neural network model is then constructed and trained on the training data.
- Using the trained model, predictions are made for the test set, 
    as well as for a specific date (30th of November in this case).
- The actual and predicted closing prices are visualized in a plot, 
    enabling direct comparison and assessment of the model's performance.
- A pseudo confidence score is provided,
    offering a simplistic representation of the model's reliability.

In the Regression_model.py there is regression model:
- This is a linear regression model is used to predict Tesla's stock closing prices based on dates.
- Converts dates to a format usable by the model.
- Splits the dataset into input (date) and output (closing price).
- Divides the dataset into training and test sets.
- Trains a linear regression model on the training data.
- Uses the model to make predictions on the test set and calculate error and R2 value to evaluate model performance.
- Creates a function that takes a date as input and returns the predicted closing price of Tesla's stock on that date, 
    along with the model's accuracy in percentage.
- Uses the model to make predictions on the entire dataset.
- Plots the actual and predicted closing prices to visualize the model's performance.

In the SERIMAX_model.py, you can find the SERIMAX_model:
- This model employs the Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors (SARIMAX) method, 
     a statistical approach suitable for time series forecasting, to predict Tesla's stock closing prices based on historical data.
- The dataset provided is initially loaded and parsed, 
     with date columns processed and transformed to be compatible with the time series requirements.
- The model zeroes in on the 'Close' price of Tesla stocks for its forecasting, ensuring a clear focus on end-of-day valuations.
- For continuity and to handle missing values, 
     the data is resampled on a daily frequency and forward-filled, maintaining the data's integrity.
- The dataset gets divided into training and test sets. The training set excludes the latest 30 days of data, 
     while the test set comprises these 30 days, establishing a clear boundary for evaluation.
- SARIMAX parameters, both non-seasonal and seasonal, are meticulously defined. 
     These parameters are crucial for the model's behavior and performance.
- Using the SARIMAX model, it's fitted with the training data, effectively "learning" from the historical stock prices.
- Leveraging the trained SARIMAX model, predictions are made extending to a specified target date, 
     which in this case is the 30th of November.
- A visual representation in the form of a plot showcases the original data alongside the SARIMAX forecast. 
     This visualization serves as a direct comparison medium, allowing for a qualitative evaluation of the forecast against actual stock prices.
- The model provides a prediction for the closing price of Tesla stocks on the 30th of November 2023, delivering a tangible, actionable insight.
- An index of forecasted dates is also outputted, providing clarity on the range and specificity of the model's predictions.


Results:
1. LSTM:
- Predicted closing price for 30th of November: $195.41
- Confidence Score: 90%

2. Regression:
- Predicted closing price on 30th of November: $270.61
- Confidence Score: 50.60%

3. SERIMAX-model:
- Predicted closing price on 30th of November: $261.35


4. ChatGPT:
- Predicted closing price on 30th of November: $191.43
- Confidence Score: 95% 

Comparing:
1. Range of Predicted Prices: 
- The predictions range from a low of $191.43 (ChatGPT) to a high of $270.61 (Regression). 
    This is quite a wide spread, showing varying outcomes from different models.

2. Confidence Scores: 
- The LSTM and ChatGPT models both have high confidence scores (90% and 95% respectively) compared to the Regression model's 50.60%. 
    It's interesting to note that while both LSTM and ChatGPT are confident in their predictions, they predict different price outcomes.

- Model Complexity and Methodology: The LSTM and SARIMAX models are inherently more complex and suitable for time series forecasting compared to the simpler linear regression model. 
    ChatGPT's methodology is based on neural network models and a vast dataset, but its application for precise numerical predictions in stock markets is unconventional.

Average Predicted Closing Price on 30th of November: $229.70

- In conclusion, while each model offers its own unique prediction based on its algorithm and methodology, 
   the average predicted price from all models is $229.70. However, it's essential to consider other external factors and expert analyses when making investment decisions.
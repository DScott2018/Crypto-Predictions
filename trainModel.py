import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load your Bitcoin price data from a CSV file
# Replace the file path with the actual path to your data
data = pd.read_csv("path")

# You can choose other columns as needed
target_column = 'High'
bitcoin_data = data[[target_column]]

# Normalize the data to the range [0, 1] for better training
scaler = MinMaxScaler()
bitcoin_data = scaler.fit_transform(bitcoin_data)

# Define the number of time steps and features
n_steps = 30  # Adjust this as needed
n_features = 1  # Typically 1 for univariate time series

# Create sequences of data for LSTM training
X = []
y = []

for i in range(len(bitcoin_data) - n_steps):
    X.append(bitcoin_data[i : i + n_steps])
    y.append(bitcoin_data[i + n_steps])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets (e.g., 80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Ensure the data has the right shape for LSTM (batch size, time steps, features)
X_train = X_train.reshape(X_train.shape[0], n_steps, n_features)
X_test = X_test.reshape(X_test.shape[0], n_steps, n_features)

# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Evaluate the model on the test data
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Make predictions using the trained model
predictions = model.predict(X_test)

# Extend the time period for prediction
n_future = len(predictions)
future_sequence = bitcoin_data[-n_steps:].reshape(1, n_steps, n_features)  # Get the last n_steps from your dataset

# Create a list to store future predictions
future_predictions = []

# Generate predictions for the future
for _ in range(n_future):
    # Use the model to predict the next price
    next_price = model.predict(future_sequence)[0, 0]

    print(f"Step {_+1}: Predicted Price = {next_price}")
    # Append the predicted price to the list
    future_predictions.append(next_price)

    # Update the future_sequence for the next prediction
    future_sequence = np.concatenate([future_sequence[:, 1:, :], np.array([[[next_price]]])], axis=1)

# Convert future_predictions to a NumPy array and then ravel it
future_predictions = np.array(future_predictions).ravel()

# Inverse transform to get actual price values for test data
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create an array for the x-axis values for predictions
x_predictions = np.arange(train_size, train_size + len(predictions))

# Create an array for the x-axis values for future predictions
x_future_predictions = np.arange(len(bitcoin_data), len(bitcoin_data) + n_future)

# Create combined_data before the plotting section
combined_data = np.full((len(bitcoin_data) + n_future, 3), np.nan)
combined_data[:len(bitcoin_data), 0] = bitcoin_data.ravel()
combined_data[train_size:train_size + len(predictions), 1] = predictions.ravel()
combined_data[len(bitcoin_data):, 2] = future_predictions.ravel()

# Plot all the data on a single graph
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(bitcoin_data)), combined_data[:len(bitcoin_data), 0], label='Actual')
plt.plot(x_predictions, combined_data[train_size:train_size + len(predictions), 1], label='Predicted')
plt.plot(x_future_predictions, combined_data[len(bitcoin_data):, 2], label='Future Predictions')
plt.legend()
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time Steps')
plt.ylabel('Normalized Price')
plt.show()

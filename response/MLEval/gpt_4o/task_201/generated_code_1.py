import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Example dataset
    # Here, we generate synthetic data for demonstration purposes.
    np.random.seed(0)
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.dot(X, np.random.rand(10, )) + np.random.rand(1000, )  # Linear relation with noise

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build the deep neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))  # Single output for regression

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)

    # Evaluate the model
    predictions = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, predictions)

    # Return the mean squared error as the output
    return mse

# Call the method for validation
output = method()
print("Mean Squared Error on Test Set:", output)
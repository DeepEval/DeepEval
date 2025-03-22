import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method():
    # Generate synthetic data for regression
    np.random.seed(0)
    X = np.random.rand(1000, 1) * 10  # 1000 samples, single feature
    y = 3 * X.squeeze() + np.random.randn(1000) * 2  # Linear relation with noise

    # Split data into training and testing sets
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Define the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(1,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test, verbose=0)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    return predictions, loss

# Call the method for validation
output, test_loss = method()
print("Test Loss:", test_loss)
print("Predictions:", output[:5])  # Print the first 5 predictions
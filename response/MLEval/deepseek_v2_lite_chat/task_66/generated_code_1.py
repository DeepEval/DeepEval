import numpy as np
import tensorflow as tf
from tensorflow import keras


def method():
    # Define the model
    model = keras.Sequential()

    # Hidden layer with 10 neurons
    model.add(keras.layers.Dense(10, activation='relu', input_shape=(None, 1)))

    # Output layer with 1 neuron
    model.add(keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Return the model
    return model


# Generate a simple test dataset
np.random.seed(0)
X_test = np.random.random((100, 1))
y_test = X_test + 0.5

# Call the method() function
model = method()
model.fit(X_test, y_test, epochs=10, verbose=1)

# Predict using the model
y_pred = model.predict(X_test)

# Print the predictions
print(y_pred)
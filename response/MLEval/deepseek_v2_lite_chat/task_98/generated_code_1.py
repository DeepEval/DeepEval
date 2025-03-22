import numpy as np
import tensorflow as tf
from tensorflow import keras


def method(x):
    # Define the model
    model = keras.Sequential([
        keras.layers.Dense(units=1, input_shape=[len(x[0])])
    ])

    # Compile the model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train the model
    model.fit(x, epochs=500, verbose=0)

    # Make a prediction
    output = model.predict(x)

    return output[0]

# Example usage:
# x = np.array([[1], [2], [3], [4]])  # Vector input
# result = method(x)
# print(result)
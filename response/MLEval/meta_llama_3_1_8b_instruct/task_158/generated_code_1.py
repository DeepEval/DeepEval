# Import necessary packages
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Define the encoding dimension
ENCODING_DIM = 128

def method():
    # Create a Sequential model
    model = Sequential()

    # Add a Flatten layer to flatten the MNIST image
    model.add(layers.Flatten(input_shape=(28, 28)))

    # Add a Dense layer to map the flattened image to a vector of size ENCODING_DIM
    model.add(layers.Dense(ENCODING_DIM, activation='relu'))

    return model

# Call the generated method for validation
output = method()
print(output.summary())
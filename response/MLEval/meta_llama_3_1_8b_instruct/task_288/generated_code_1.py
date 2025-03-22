from keras.layers import Dense
from keras.models import Sequential
import numpy as np


def method(num_inputs=64, num_nodes=32):
    # Create a neural network model
    model = Sequential()

    # Add the first layer with the specified number of inputs and nodes in the hidden layer
    model.add(Dense(num_nodes, activation='relu', input_shape=(num_inputs,)))

    # Return the output of the model
    return model


# Define a sample input
input_data = np.random.rand(1, 64)

# Create a neural network model with the specified number of inputs and nodes in the hidden layer
model = method(num_inputs=64, num_nodes=32)


# Generate an output by passing the input data through the model
output = model.predict(input_data)

# Print the generated output
print(output)


# Call the method() function for validation
model = method(num_inputs=64, num_nodes=32)
output = model.predict(np.random.rand(1, 64))
print(output)
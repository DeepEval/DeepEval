import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def method():
    # Creating a simple model with an extra hidden layer
    model = keras.Sequential()
    
    # Input layer (assuming input features size of 20 for this example)
    model.add(layers.InputLayer(input_shape=(20,)))
    
    # First hidden layer with a specified number of neurons
    model.add(layers.Dense(10, activation='relu'))  # 10 neurons in hidden layer
    
    # Output layer with a single neuron
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification output
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # You would typically train the model here with input data and labels
    # For demonstration, we're returning the model summary as 'output'
    output = model.summary()
    
    return output

# Call the method to validate it
if __name__ == "__main__":
    method_output = method()
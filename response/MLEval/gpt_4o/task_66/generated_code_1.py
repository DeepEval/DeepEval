import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Create a simple Sequential model
    model = Sequential()
    
    # Add a hidden layer with 10 neurons
    model.add(Dense(10, activation='relu', input_shape=(input_shape,)))
    
    # Add another hidden layer with 10 neurons
    model.add(Dense(10, activation='relu'))
    
    # Add the output layer with a single neuron
    model.add(Dense(1, activation='sigmoid'))  # Assuming a binary classification task

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Call the method to validate it
output_model = method()

# Print a summary of the model to validate the structure
output_model.summary()
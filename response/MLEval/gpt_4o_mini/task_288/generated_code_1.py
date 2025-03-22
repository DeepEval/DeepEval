import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method(num_inputs, num_nodes):
    # Create a Sequential model
    model = keras.Sequential()
    
    # Add the first layer (hidden layer)
    model.add(layers.Dense(num_nodes, input_shape=(num_inputs,), activation='relu'))
    
    # Optionally, you can add an output layer if needed
    # For example, if you have 3 classes, you might do:
    # model.add(layers.Dense(3, activation='softmax'))
    
    # Compile the model (this step is optional until you are ready to train)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example of calling the method
num_inputs = 10  # Replace with your number of inputs
num_nodes = 5    # Replace with your desired number of nodes in the hidden layer
output = method(num_inputs, num_nodes)

# To validate the output, we can print the model summary
output.summary()
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method():
    # Initialize a sequential model
    model = Sequential()
    
    # Add an input layer with a specified number of input dimensions (e.g., 10)
    model.add(Dense(10, input_dim=10, activation='relu'))
    
    # Add an additional hidden layer with 6 nodes
    model.add(Dense(6, activation='relu'))
    
    # Add an output layer (e.g., for binary classification, use 1 node with sigmoid activation)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Return the model as output
    return model

# Call the method for validation
output_model = method()
output_model.summary()
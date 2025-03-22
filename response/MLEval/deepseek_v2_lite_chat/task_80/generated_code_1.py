import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def method(input_shape=(1,), num_classes=10):
    # Create a Sequential model
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape),  # First dense layer
        Dense(64, activation='relu'),  # Second dense layer
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    
    # Add a new layer after the second dense layer
    model.add(Dense(64, activation='relu'))  # New dense layer
    model.add(Dense(num_classes, activation='softmax'))  # New output layer
    
    return model

# Example input shape
input_shape = (100,)
num_classes = 10

# Create and print the model
model = method(input_shape=input_shape, num_classes=num_classes)
model.summary()
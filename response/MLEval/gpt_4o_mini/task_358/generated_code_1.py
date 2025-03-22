import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def method():
    # Define the model
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(28, 28, 1)))  # Example input shape for grayscale images (like MNIST)
    
    # Hidden layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Another convolutional layer
    model.add(layers.MaxPooling2D((2, 2)))  # Another max pooling layer
    model.add(layers.Flatten())  # Flattening the output from 2D to 1D
    model.add(layers.Dense(64, activation='relu'))  # Fully connected layer
    model.add(layers.Dense(10, activation='softmax'))  # Output layer for classification (10 classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()
    
    # Return the model for further use
    return model

# Call the method to create the model and validate
output_model = method()
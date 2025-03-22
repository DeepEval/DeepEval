# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def method():
    """
    Build a sequential model for machine learning tasks.
    
    The model will consist of several layers including convolutional, 
    pooling, and dense layers. Dropout will be used to reduce overfitting.
    """
    
    # Define the sequential model
    model = keras.Sequential()
    
    # Convolutional layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    
    # Convolutional layer 2
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    
    # Pooling layer 1
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Convolutional layer 3
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Convolutional layer 4
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Pooling layer 2
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten layer
    model.add(layers.Flatten())
    
    # Dense layer 1
    model.add(layers.Dense(64, activation='relu'))
    
    # Dropout to reduce overfitting
    model.add(layers.Dropout(0.2))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Call the method for validation
model = method()
print(model.summary())
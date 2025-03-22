# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    # Define the input shape of the images
    input_shape = (32, 32, 3)
    
    # Define the inputs
    inputs = keras.Input(shape=input_shape)
    
    # Define the first convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Add the output of the first convolutional layer to the output of the third convolutional layer
    x_1 = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x_2 = layers.Conv2D(64, (3, 3), activation='relu')(x_1)
    x_3 = layers.Conv2D(64, (3, 3), activation='relu')(x_2)
    x_add_1 = layers.Add()([x_1, x_3])
    
    # Define the second convolutional layer that processes the input directly
    x_4 = layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    
    # Add the added outputs from all paths
    x_add = layers.Add()([x_add_1, x_4])
    
    # Flatten the output
    x = layers.Flatten()(x_add)
    
    # Define the fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the function
model = dl_model()
model.summary()
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

def dl_model():
    # Input layer for 28x28 images with a single channel (grayscale)
    inputs = layers.Input(shape=(28, 28, 1))
    
    # First 1x1 convolution to reduce dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(inputs)
    
    # 3x3 convolutional layer for feature extraction
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
    
    # Second 1x1 convolution to restore dimensionality
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Fully connected layer with 10 neurons for classification
    outputs = layers.Dense(units=10, activation='softmax')(x)
    
    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# To create the model
model = dl_model()
model.summary()  # This will print the model architecture
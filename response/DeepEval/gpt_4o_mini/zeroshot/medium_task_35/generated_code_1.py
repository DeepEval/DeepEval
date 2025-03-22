import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))
    
    # Stage 1: Convolution and Max Pooling
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Stage 2: Convolution and Max Pooling
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Additional Convolutional layers
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    
    # UpSampling with Skip Connections
    x_skip1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x_skip1 = layers.UpSampling2D((2, 2))(x_skip1)

    # Skip connection from the second stage
    x_skip1 = layers.Concatenate()([x_skip1, x])  # Concatenate with the second stage output

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x_skip1)
    
    x_skip2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x_skip2 = layers.UpSampling2D((2, 2))(x_skip2)

    # Skip connection from the first stage
    x_skip2 = layers.Concatenate()([x_skip2, inputs])  # Concatenate with the input layer

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x_skip2)
    
    # Final output layer with 1x1 Convolution
    outputs = layers.Conv2D(10, (1, 1), activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs, outputs)
    
    return model

# Example of how to compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

def dl_model():
    """
    This function generates a deep learning model for image classification using CIFAR-10 dataset.
    
    The model consists of three parts:
    1. Feature extraction through 3 pairs of convolutional and max-pooling layers.
    2. Processing through 3 convolutional, dropout, and convolutional layers to enhance generalization.
    3. Upsampling through 3 pairs of convolutional and transposed convolutional layers with skip connections.
    4. Final classification through a 1x1 convolutional layer.
    """
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = keras.Input(shape=input_shape)
    
    # Part 1: Feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Part 2: Generalization enhancement
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    
    # Part 3: Upsampling and spatial restoration
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(x)
    x = layers.Concatenate()([x, layers.Conv2D(64, (3, 3), activation='relu', padding='same')(layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(x))])
    x = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu')(x)
    x = layers.Concatenate()([x, layers.Conv2D(32, (3, 3), activation='relu', padding='same')(layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu')(x))])
    x = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), activation='relu')(x)
    x = layers.Concatenate()([x, layers.Conv2D(16, (3, 3), activation='relu', padding='same')(layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), activation='relu')(x))])
    
    # Final classification
    x = layers.Conv2D(10, (1, 1), activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=x)
    
    return model
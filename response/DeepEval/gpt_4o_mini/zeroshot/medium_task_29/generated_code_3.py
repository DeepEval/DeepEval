import numpy as np
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate
from keras.utils import to_categorical

def dl_model():
    # Input layer for 32x32 RGB images (CIFAR-10)
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layers (for feature extraction)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # First Max Pooling Layer (1x1)
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    flattened1 = Flatten()(pooling1)
    
    # Second Max Pooling Layer (2x2)
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    flattened2 = Flatten()(pooling2)
    
    # Third Max Pooling Layer (4x4)
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    flattened3 = Flatten()(pooling3)
    
    # Concatenate the flattened features
    concatenated = Concatenate()([flattened1, flattened2, flattened3])
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(concatenated)
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Output layer for 10 classes with softmax activation
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Creating the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage: Initialize the model
model = dl_model()
model.summary()
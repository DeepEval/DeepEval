import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    inputs = keras.Input(shape=(28, 28, 1)) # Input shape for MNIST

    # Block 1: Parallel Average Pooling
    x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    x3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)

    x1 = layers.Flatten()(x1)
    x2 = layers.Flatten()(x2)
    x3 = layers.Flatten()(x3)
    
    x = layers.Concatenate()([x1, x2, x3])

    x = layers.Dense(128, activation='relu')(x)  # Fully connected layer
    x = layers.Reshape((1, 128))(x)  # Reshape for Block 2

    # Block 2: Feature Extraction Branches
    branch1 = layers.Sequential([
        layers.Conv2D(32, kernel_size=(1, 1), activation='relu'),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
    ])(x)

    branch2 = layers.Sequential([
        layers.Conv2D(32, kernel_size=(1, 1), activation='relu'),
        layers.Conv2D(64, kernel_size=(1, 7), activation='relu'),
        layers.Conv2D(64, kernel_size=(7, 1), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
    ])(x)

    branch3 = layers.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
    ])(x)

    x = layers.Concatenate()([branch1, branch2, branch3])

    # Output Layers
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x) # Output layer for 10 classes

    model = keras.Model(inputs=inputs, outputs=x)
    return model
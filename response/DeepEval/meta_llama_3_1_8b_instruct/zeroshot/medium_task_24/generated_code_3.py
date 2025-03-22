# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Branch 1: 1x1 convolution and 3x3 convolution
    x1 = layers.Conv2D(32, 1, activation='relu')(inputs)
    x1 = layers.Conv2D(32, 3, activation='relu')(x1)
    x1 = layers.Dropout(0.2)(x1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    x2 = layers.Conv2D(32, 1, activation='relu')(inputs)
    x2 = layers.Conv2D(32, 1, activation='relu')(x2)
    x2 = layers.Conv2D(32, 7, activation='relu')(x2)
    x2 = layers.Conv2D(32, 7, activation='relu')(x2)
    x2 = layers.Conv2D(32, 3, activation='relu')(x2)
    x2 = layers.Dropout(0.2)(x2)

    # Branch 3: Max pooling
    x3 = layers.MaxPooling2D((2, 2))(inputs)
    x3 = layers.Conv2D(32, 3, activation='relu')(x3)
    x3 = layers.Dropout(0.2)(x3)

    # Concatenate the outputs from all branches
    concatenated = layers.Concatenate()([x1, x2, x3])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated)

    # Add fully connected layers
    x = layers.Dense(128, activation='relu')(flattened)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model
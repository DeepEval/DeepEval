import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 3x3 Convolution
    branch1 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    branch1 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch1)

    # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
    branch2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: Max Pooling
    branch3 = layers.MaxPooling2D(pool_size=(2, 2))(inputs)

    # Concatenate the outputs from the three branches
    concatenated = layers.concatenate([branch1, branch2, branch3], axis=-1)

    # Flatten the concatenated features
    flattened = layers.Flatten()(concatenated)

    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened)
    dense1 = layers.Dropout(0.5)(dense1)  # Dropout layer for regularization
    dense2 = layers.Dense(10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
model = dl_model()
model.summary()
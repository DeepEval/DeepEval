import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_shape = (28, 28, 1)  # MNIST images are 28x28 pixels and have 1 channel (grayscale)
    inputs = layers.Input(shape=input_shape)

    # Block 1 - Primary Path
    x1 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(inputs)
    x1 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(x1)
    x1 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x1)

    # Block 1 - Branch Path
    x2 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x2 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x2)

    # Concatenate both paths
    x = layers.concatenate([x1, x2], axis=-1)  # Concatenate along the channel dimension

    # Block 2 - Reshape and Channel Shuffle
    shape_before_reshape = x.shape[1:]  # Get the shape of the features (height, width, channels)
    groups = 4
    channels_per_group = shape_before_reshape[-1] // groups

    # Reshape into (height, width, groups, channels_per_group)
    x = layers.Reshape((shape_before_reshape[0], shape_before_reshape[1], groups, channels_per_group))(x)
    
    # Permute dimensions to achieve channel shuffle
    x = layers.Permute((0, 1, 3, 2))(x)  # Swap groups and channels

    # Reshape back to original shape
    x = layers.Reshape((shape_before_reshape[0], shape_before_reshape[1], -1))(x)

    # Fully connected layer for classification
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for MNIST

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()
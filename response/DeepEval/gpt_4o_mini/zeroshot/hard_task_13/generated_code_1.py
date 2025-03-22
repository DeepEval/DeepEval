import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32x3)
    inputs = layers.Input(shape=(32, 32, 3))

    # Parallel branches
    branch1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    branch3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(inputs)
    branch4 = layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)

    # Concatenate the outputs of the branches
    concat = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Second block: Global Average Pooling
    global_avg_pool = layers.GlobalAveragePooling2D()(concat)

    # Fully connected layers
    fc1 = layers.Dense(128, activation='relu')(global_avg_pool)
    fc2 = layers.Dense(128, activation='relu')(fc1)

    # Create weights for the reshape operation
    weights = layers.Dense(concat.shape[-1])(fc2)  # Generates weights of the same size as channels in concat
    weights_reshaped = layers.Reshape((1, 1, concat.shape[-1]))(weights)  # Reshape to match input feature map

    # Element-wise multiplication
    multiplied = layers.multiply([concat, weights_reshaped])

    # Final fully connected layer to produce output probabilities
    flattened = layers.GlobalAveragePooling2D()(multiplied)
    output = layers.Dense(10, activation='softmax')(flattened)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=output)

    return model

# Example of how to create the model
model = dl_model()
model.summary()
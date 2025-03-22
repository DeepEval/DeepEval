import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Path 1: 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Path 2: Sequence of convolutions (1x1, 1x7, 7x1)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path2 = layers.Conv2D(32, (1, 7), activation='relu')(path2)
    path2 = layers.Conv2D(32, (7, 1), activation='relu')(path2)

    # Concatenate the outputs of both paths
    concatenated = layers.Concatenate()([path1, path2])

    # 1x1 Convolution to align the output dimensions with the input channels
    main_output = layers.Conv2D(32, (1, 1), activation='relu')(concatenated)

    # Add the branch that connects directly to the input
    merged = layers.Add()([main_output, inputs])

    # Global Average Pooling to reduce the spatial dimensions
    pooled = layers.GlobalAveragePooling2D()(merged)

    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(pooled)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=x)

    return model

# Create the model
model = dl_model()
model.summary()
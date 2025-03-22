import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 pixels with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Initial Convolutional Layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)

    # Global Average Pooling Path
    avg_pooling = layers.GlobalAveragePooling2D()(x)
    avg_dense1 = layers.Dense(128, activation='relu')(avg_pooling)
    avg_dense2 = layers.Dense(128, activation='relu')(avg_dense1)

    # Global Max Pooling Path
    max_pooling = layers.GlobalMaxPooling2D()(x)
    max_dense1 = layers.Dense(128, activation='relu')(max_pooling)
    max_dense2 = layers.Dense(128, activation='relu')(max_dense1)

    # Channel Attention Weights
    channel_features = layers.Add()([avg_dense2, max_dense2])
    channel_attention = layers.Activation('sigmoid')(channel_features)

    # Apply channel attention
    channel_refined = layers.Multiply()([x, layers.Reshape((1, 1, 128))(channel_attention)])

    # Spatial Features Extraction
    avg_spatial = layers.GlobalAveragePooling2D()(channel_refined)
    max_spatial = layers.GlobalMaxPooling2D()(channel_refined)

    # Concatenate Spatial Features
    spatial_features = layers.Concatenate()([avg_spatial, max_spatial])

    # Combine Channel and Spatial Features
    combined_features = layers.Multiply()([channel_refined, layers.Reshape((1, 1, 128))(spatial_features)])

    # Flatten and Fully Connected Layer
    flattened = layers.Flatten()(combined_features)
    outputs = layers.Dense(10, activation='softmax')(flattened)  # CIFAR-10 has 10 classes

    # Create the Model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of model creation
model = dl_model()
model.summary()
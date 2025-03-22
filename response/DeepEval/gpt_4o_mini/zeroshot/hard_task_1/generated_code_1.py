import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model(input_shape=(32, 32, 3), num_classes=10):
    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Initial Convolutional Layer
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)

    # Block 1: Parallel processing paths
    # Path 1: Global Average Pooling and Fully Connected Layers
    path1 = layers.GlobalAveragePooling2D()(x)
    path1 = layers.Dense(64, activation='relu')(path1)
    path1 = layers.Dense(32, activation='relu')(path1)

    # Path 2: Global Max Pooling and Fully Connected Layers
    path2 = layers.GlobalMaxPooling2D()(x)
    path2 = layers.Dense(64, activation='relu')(path2)
    path2 = layers.Dense(32, activation='relu')(path2)

    # Channel Attention weights
    channel_attention = layers.Add()([path1, path2])
    channel_attention = layers.Activation('sigmoid')(channel_attention)

    # Element-wise multiplication with original features
    channel_attention = layers.Reshape((1, 1, 32))(channel_attention)  # Reshape to match the feature map
    x = layers.multiply([x, channel_attention])

    # Block 2: Spatial features extraction
    avg_pool = layers.AveragePooling2D(pool_size=(2, 2))(x)
    max_pool = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Concatenate along the channel dimension
    spatial_features = layers.Concatenate()([avg_pool, max_pool])
    
    # 1x1 Convolution and Sigmoid activation to normalize features
    spatial_features = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='sigmoid')(spatial_features)

    # Element-wise multiplication with channel attention features
    x = layers.multiply([x, spatial_features])

    # Additional branch with 1x1 convolution to align output channels
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(x)

    # Add to the main path
    x = layers.Add()([x, inputs])
    x = layers.Activation('relu')(x)

    # Global Average Pooling followed by a Fully Connected layer for classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs, x)

    return model

# Example of creating the model
model = dl_model()
model.summary()
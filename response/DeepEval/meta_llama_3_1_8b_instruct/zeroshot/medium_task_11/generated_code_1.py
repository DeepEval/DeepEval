import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer with shape (32, 32, 3) for CIFAR-10 dataset
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)

    # Feature extraction using two parallel paths
    # Path 1: Global average pooling followed by two fully connected layers
    path1 = layers.GlobalAveragePooling2D()(x)
    path1 = layers.Dense(128, activation='relu')(path1)
    path1 = layers.Dense(10, activation='softmax')(path1)

    # Path 2: Global max pooling followed by two fully connected layers
    path2 = layers.GlobalMaxPooling2D()(x)
    path2 = layers.Dense(128, activation='relu')(path2)
    path2 = layers.Dense(10, activation='softmax')(path2)

    # Channel attention weights
    channel_weights = layers.Add()([path1, path2])
    channel_weights = layers.Activation('softmax')(channel_weights)

    # Apply channel attention weights to the original features
    channel_features = layers.Multiply()([x, channel_weights])

    # Separate average and max pooling operations to extract spatial features
    avg_pool = layers.GlobalAveragePooling2D()(channel_features)
    max_pool = layers.GlobalMaxPooling2D()(channel_features)

    # Concatenate spatial features along the channel dimension
    spatial_features = layers.Concatenate()([avg_pool, max_pool])

    # Combine channel features with spatial features
    combined_features = layers.Multiply()([channel_features, spatial_features])

    # Flatten and feed into a fully connected layer
    outputs = layers.Flatten()(combined_features)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Test the model
model = dl_model()
model.summary()
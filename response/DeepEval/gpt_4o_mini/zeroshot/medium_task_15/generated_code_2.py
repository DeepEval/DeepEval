import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Convolutional layer
    conv = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    conv = layers.BatchNormalization()(conv)
    conv = layers.Activation('relu')(conv)

    # Global average pooling
    gap = layers.GlobalAveragePooling2D()(conv)

    # Fully connected layers
    fc1 = layers.Dense(32, activation='relu')(gap)
    fc2 = layers.Dense(32, activation='relu')(fc1)

    # Reshape and multiply with initial features
    reshaped_fc = layers.Reshape((1, 1, 32))(fc2)
    weighted_features = layers.multiply([reshaped_fc, conv])

    # Concatenate with the input layer
    concatenated = layers.concatenate([inputs, weighted_features])

    # Downsampling with 1x1 convolution and average pooling
    conv1x1 = layers.Conv2D(16, (1, 1), padding='same')(concatenated)
    pooled = layers.AveragePooling2D(pool_size=(2, 2))(conv1x1)

    # Final fully connected layer for classification
    flat = layers.Flatten()(pooled)
    outputs = layers.Dense(10, activation='softmax')(flat)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of creating the model
model = dl_model()
model.summary()
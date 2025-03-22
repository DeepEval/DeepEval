import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the image data
    inputs = layers.Input(shape=(32, 32, 3))

    # First max pooling layer
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)

    # Second max pooling layer
    y = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)

    # Third max pooling layer
    z = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    z = layers.Conv2D(128, (3, 3), activation='relu')(z)

    # Flatten the output from each pooling layer
    x_flat = layers.Flatten()(x)
    y_flat = layers.Flatten()(y)
    z_flat = layers.Flatten()(z)

    # Concatenate the flattened vectors
    concat = layers.concatenate([x_flat, y_flat, z_flat])

    # Fully connected layer 1
    dense_1 = layers.Dense(64, activation='relu')(concat)

    # Fully connected layer 2 for classification
    outputs = layers.Dense(10, activation='softmax')(dense_1)

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model
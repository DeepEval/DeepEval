import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # First block
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.AveragePooling2D()(x)

    # Main path
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(x)
    main_path = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    main_path = layers.AveragePooling2D()(main_path)

    # Skip connection
    skip_path = layers.Conv2D(64, (1, 1), activation='relu')(x)
    skip_path = layers.AveragePooling2D()(skip_path)

    # Second block
    y = layers.Conv2D(64, (3, 3), activation='relu')(main_path)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)

    # Channel weights
    channel_weights = layers.GlobalAveragePooling2D()(y)
    channel_weights = layers.Dense(32, activation='relu')(channel_weights)
    channel_weights = layers.Dense(64, activation='sigmoid')(channel_weights)

    # Refined weights
    refined_weights = layers.Reshape((1, 1, 64))(channel_weights)
    refined_weights = layers.Multiply()([refined_weights, y])

    # Output
    outputs = layers.Flatten()(refined_weights)
    outputs = layers.Dense(10, activation='softmax')(outputs)

    # Model definition
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
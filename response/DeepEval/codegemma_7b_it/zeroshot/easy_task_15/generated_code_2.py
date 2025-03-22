import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    # Input layer
    inputs = keras.Input(shape=(28, 28, 1))

    # Specialized block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Second specialized block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Flatten and fully connected
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Model construction
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
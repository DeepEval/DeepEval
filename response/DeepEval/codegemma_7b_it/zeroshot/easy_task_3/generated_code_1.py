import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Feature extraction
    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Feature flattening
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
model = dl_model()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x_main = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x_main = layers.Conv2D(64, (3, 3), activation='relu')(x_main)
    x_main = layers.MaxPooling2D((2, 2))(x_main)

    # Branch Path
    x_branch = layers.Conv2D(16, (5, 5), activation='relu')(inputs)

    # Concatenate features
    x = layers.concatenate([x_main, x_branch])

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Main Path & Branch Path
    x_main = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x_main = layers.Dropout(0.2)(x_main)
    x_main = layers.Conv2D(3, (3, 3), activation='relu', padding='same')(x_main)

    x_branch = inputs

    x = layers.add([x_main, x_branch])

    # Block 2: Separable Convolutions
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)  

    # Group 1
    x1 = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(x[0])
    x1 = layers.Dropout(0.2)(x1)

    # Group 2
    x2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x[1])
    x2 = layers.Dropout(0.2)(x2)

    # Group 3
    x3 = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x[2])
    x3 = layers.Dropout(0.2)(x3)

    x = layers.concatenate([x1, x2, x3], axis=-1)

    # Flatten and Output
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x) 

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model
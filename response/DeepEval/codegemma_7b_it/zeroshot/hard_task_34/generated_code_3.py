import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    inputs = keras.Input(shape=(28, 28, 1))

    # Main Path
    x = inputs
    for _ in range(3):
        x = layers.ReLU()(x)
        x = layers.SeparableConv2D(filters=32, kernel_size=3, padding='same')(x)
        x = layers.MaxPooling2D(pool_size=2)(x)

    # Branch Path
    y = layers.Conv2D(filters=32, kernel_size=3, padding='same')(inputs)
    y = layers.MaxPooling2D(pool_size=2)(y)

    # Fusion
    x = layers.concatenate([x, y])
    x = layers.Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

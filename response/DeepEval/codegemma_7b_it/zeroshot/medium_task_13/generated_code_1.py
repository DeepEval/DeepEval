import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = inputs
    for _ in range(3):
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.concatenate([x, inputs])  # Concatenate with input along channels
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage:
model = dl_model()
model.summary()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Block 2
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x = layers.SeparableConv2D(16, (1, 1), activation='relu')(x[0])
    x = layers.Dropout(0.2)(x)
    x = layers.SeparableConv2D(16, (3, 3), activation='relu')(x[1])
    x = layers.Dropout(0.2)(x)
    x = layers.SeparableConv2D(16, (5, 5), activation='relu')(x[2])
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Model
    model = keras.Model(inputs=inputs, outputs=outputs, name='cifar10_model')

    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
    shortcut = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(inputs)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    # Block 2
    x = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Block 3
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = [
        layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(v)
        for v in x
    ]
    x = [
        layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(v)
        for v in x
    ]
    x = [
        layers.Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(v)
        for v in x
    ]
    x = [layers.Dropout(rate=0.2)(v) for v in x]

    # Block 4
    x = layers.concatenate(x)
    x = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Output layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1:
    x = layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same')(inputs)
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    x = [
        layers.Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(xi)
        for xi in x
    ]
    x = layers.Concatenate(axis=3)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2:
    x = layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same')(x)
    branches = [
        [x, layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')],
        [x, layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')],
        [x, layers.Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')],
        [x, layers.MaxPooling2D(pool_size=(3, 3), padding='same'), layers.Conv2D(filters=128, kernel_size=(1, 1), padding='same', activation='relu')],
    ]
    outputs = []
    for branch in branches:
        y = branch[0]
        for layer in branch[1:]:
            y = layer(y)
        outputs.append(y)

    x = layers.Concatenate(axis=3)(outputs)

    # Output layer:
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    return model
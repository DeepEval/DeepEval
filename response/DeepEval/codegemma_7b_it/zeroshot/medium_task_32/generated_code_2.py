import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Split the input into three groups
    group1 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    group2 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    group3 = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Feature extraction for each group
    group1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same')(group1)
    group2 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(group2)
    group3 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same')(group3)

    # Concatenate and fuse features
    concat = layers.concatenate([group1, group2, group3])

    # Flatten and classify
    flatten = layers.Flatten()(concat)
    outputs = layers.Dense(10, activation='softmax')(flatten)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
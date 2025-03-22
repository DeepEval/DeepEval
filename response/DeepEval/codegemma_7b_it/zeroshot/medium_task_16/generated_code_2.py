import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    inputs = keras.Input(shape=(32, 32, 3))

    x = tf.split(inputs, num_or_size_splits=3, axis=3)
    x = [layers.Conv2D(filters=inputs.shape[-1] // 3, kernel_size=1, padding='same')(xi) for xi in x]
    x = [layers.AveragePooling2D(pool_size=2, strides=2, padding='valid')(xi) for xi in x]

    concat = layers.Concatenate(axis=3)(x)

    flatten = layers.Flatten()(concat)
    dense1 = layers.Dense(units=64, activation='relu')(flatten)
    outputs = layers.Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
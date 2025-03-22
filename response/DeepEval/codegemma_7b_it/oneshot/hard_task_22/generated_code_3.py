import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    inputs = keras.Input(shape=(32, 32, 3))

    # Main Path
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    group_1 = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    group_1 = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')
    ])(group_1)

    group_2 = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    group_2 = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')
    ])(group_2)

    group_3 = tf.keras.layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    group_3 = keras.Sequential([
        layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same')
    ])(group_3)

    concat = keras.layers.Concatenate()([group_1, group_2, group_3])

    x = layers.BatchNormalization()(concat)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Branch Path
    y = layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(inputs)
    y = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(y)

    y = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(y)

    # Fusion
    fused = layers.Add()([x, y])
    fused = layers.Activation('relu')(fused)

    flatten = layers.Flatten()(fused)
    outputs = layers.Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
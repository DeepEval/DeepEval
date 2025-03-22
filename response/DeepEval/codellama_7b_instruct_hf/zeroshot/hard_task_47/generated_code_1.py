import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # First block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x = layers.Conv2D(32, (1, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (5, 5), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, x[1], x[2]])

    # Second block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x = layers.Conv2D(64, (1, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (1, 7), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (7, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, x[1], x[2]])

    # Third block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x = layers.Conv2D(128, (1, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (1, 7), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (7, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, x[1], x[2]])

    # Final block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x = layers.Conv2D(256, (1, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (1, 7), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (7, 1), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), padding='same', depthwise_separable=True)(x[0])
    x = layers.BatchNormalization()(x)
    x = layers.Concatenate()([x, x[1], x[2]])

    # Output block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    return model
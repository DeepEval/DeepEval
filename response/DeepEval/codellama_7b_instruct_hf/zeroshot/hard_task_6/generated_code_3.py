import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x[0])
    x = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x[1])
    x = layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x[2])
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=3, activation='relu')(x)
    x = layers.Conv2D(filters=512, kernel_size=3, activation='relu')(x)
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    return model

model = dl_model()
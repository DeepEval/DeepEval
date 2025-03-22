import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_block(x, filters):
    path = keras.Sequential([
        layers.BatchNormalization(),
        layers.SeparableConv2D(filters, (1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(filters, (3, 3), padding='same'),
    ])(x)
    return path

def path1(x):
    return conv_block(x, 96)

def path2(x):
    x = layers.AveragePooling2D((3, 3), strides=(2, 2), padding='same')(x)
    return conv_block(x, 96)

def path3(x):
    x = conv_block(x, 64)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(64, (1, 3), padding='same')(x)
    path = layers.BatchNormalization()(x)
    path = layers.SeparableConv2D(96, (3, 1), padding='same')(path)
    return path

def path4(x):
    x = conv_block(x, 64)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(64, (1, 3), padding='same')(x)
    path = layers.BatchNormalization()(x)
    path = layers.SeparableConv2D(96, (3, 1), padding='same')(path)
    return path

def block(x):
    path = tf.split(x, num_or_size_splits=3, axis=3)
    path1 = path2 = path3 = path4 = x
    path1 = path1(path1)
    path2 = path2(path2)
    path3 = path3(path3)
    path4 = path4(path4)
    path = tf.concat([path1, path2, path3, path4], axis=3)
    path = layers.BatchNormalization()(path)
    path = layers.SeparableConv2D(96, (3, 3), padding='same')(path)
    return path

def classification_model():
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = block(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)
    x = block(x)
    x = layers.BatchNormalization()(x)
    x = layers.SeparableConv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(units=10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = classification_model()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3))

    # Extract initial features using convolutional, batch normalization, and ReLU activation
    x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Compress feature maps using global average pooling and two fully connected layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(units=32, activation='relu')(x)

    # Reshape, generate weighted feature maps, concatenate with input layer
    x = layers.Reshape((32, 32, 32))(x)
    weighted_features = layers.Multiply()([x, inputs])
    x = layers.Concatenate()([weighted_features, inputs])

    # Reduce dimensionality and downsample features using 1x1 convolution and average pooling
    x = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    # Final fully connected layer for classification
    outputs = layers.Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
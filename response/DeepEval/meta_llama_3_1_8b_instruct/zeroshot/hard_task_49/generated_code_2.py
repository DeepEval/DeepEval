# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # First block: average pooling layers
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(inputs)
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=1, padding='same')(x)
    x = layers.AveragePooling2D(pool_size=(8, 8), strides=1, padding='same')(x)
    x = layers.Flatten()(x)

    # Concatenate the flattened vectors
    x = layers.Concatenate()([layers.Flatten()(layers.AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(inputs)),
                              layers.Flatten()(layers.AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(inputs))])

    # Fully connected layer and reshape
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Reshape((4, 128))(x)

    # Second block: depthwise separable convolutional layers
    x = tf.split(x, 4, axis=1)
    x = [layers.SeparableConv2D(32, kernel_size=(1, 1), activation='relu')(i) for i in x]
    x = [layers.SeparableConv2D(64, kernel_size=(3, 3), activation='relu')(i) for i in x]
    x = [layers.SeparableConv2D(128, kernel_size=(5, 5), activation='relu')(i) for i in x]
    x = [layers.SeparableConv2D(256, kernel_size=(7, 7), activation='relu')(i) for i in x]
    x = layers.Concatenate()(x)

    # Flatten and fully connected layer
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
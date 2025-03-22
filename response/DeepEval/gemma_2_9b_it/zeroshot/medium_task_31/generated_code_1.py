import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(32, 32, 3)))

    # Split the channels
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=2)))

    # Apply different convolutional kernels to each channel group
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))  # 1x1 kernels
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))  # 3x3 kernels
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))  # 5x5 kernels

    # Concatenate the outputs
    model.add(layers.Concatenate(axis=2))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax')) 

    return model
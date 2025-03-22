import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    model = keras.Sequential(name='mnist_model')

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='block1_conv1'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', name='block1_conv2'))
    model.add(layers.MaxPooling2D((2, 2), name='block1_max_pool'))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='block2_conv1'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='block2_conv2'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', name='block2_conv3'))
    model.add(layers.MaxPooling2D((2, 2), name='block2_max_pool'))

    model.add(layers.Flatten(name='flatten'))

    model.add(layers.Dense(64, activation='relu', name='dense_1'))
    model.add(layers.Dense(64, activation='relu', name='dense_2'))
    model.add(layers.Dense(10, activation='softmax', name='dense_3'))

    return model
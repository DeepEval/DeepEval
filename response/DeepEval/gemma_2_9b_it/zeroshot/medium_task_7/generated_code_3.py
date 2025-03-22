import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    model = tf.keras.models.Sequential()

    # Input Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    # Path 1
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    # Path 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Combine outputs
    model.add(layers.Add() )

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model
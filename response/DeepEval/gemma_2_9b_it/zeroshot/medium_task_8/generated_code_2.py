import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    
    model = tf.keras.Sequential()

    # Main Path
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1), input_shape=(32, 32, 3)))
    model.add(layers.Lambda(lambda x: x[0]))  # First group unchanged
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Second group
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Combined third and second
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=-1)))  # Concatenate all groups

    # Branch Path
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))  

    # Fusion
    model.add(layers.Add() )  

    # Classification
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax')) 

    return model
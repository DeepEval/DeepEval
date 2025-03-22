import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential([
        layers.Input(shape=(28, 28, 1)),  
        layers.Conv2D(filters=32, kernel_size=1, activation='relu'),
        layers.DepthwiseConv2D(kernel_size=3, activation='relu', padding='same'),
        layers.Conv2D(filters=32, kernel_size=1, activation='relu'),
        layers.Add() ,  # Add residual connection
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model
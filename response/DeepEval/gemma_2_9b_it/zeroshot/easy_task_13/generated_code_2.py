import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(28, 28, 1)))  

    # 1x1 Convolutional Layers
    model.add(layers.Conv1D(filters=32, kernel_size=1, activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))  
    model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))  

    # 3x1 Convolutional Layer
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Dropout(0.3))

    # 1x3 Convolutional Layer
    model.add(layers.Conv1D(filters=64, kernel_size=1, activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))

    # Restore channels
    model.add(layers.Conv1D(filters=1, kernel_size=1, activation='sigmoid', padding='same'))

    # Feature Combination
    model.add(layers.Add() ) 

    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model
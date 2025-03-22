import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()
    
    # Input layer
    model.add(layers.Input(shape=(32, 32, 3))) 

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D()) 

    # Dense layer 1
    model.add(layers.Dense(32, activation='relu'))

    # Dense layer 2
    model.add(layers.Dense(32, activation='relu'))

    # Reshape and element-wise multiplication
    model.add(layers.Reshape((32, 32, 3)))
    model.add(layers.Multiply())

    # Flatten and final dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model
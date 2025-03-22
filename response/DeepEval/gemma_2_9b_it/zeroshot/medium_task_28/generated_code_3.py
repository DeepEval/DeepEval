import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Attention Layer
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1), activation='softmax', input_shape=(32, 32, 3)))  
    # Apply attention weights
    model.add(layers.Multiply() )
    
    # Dimensionality Reduction
    model.add(layers.Conv2D(filters=int(32*1/3), kernel_size=(1, 1)))
    model.add(layers.LayerNormalization())
    model.add(layers.ReLU())

    # Restore Dimensionality
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1)))
    
    # Add Contextual Information
    model.add(layers.Add() ) 
    
    # Classification Head
    model.add(layers.Flatten())
    model.add(layers.Dense(units=10, activation='softmax'))

    return model
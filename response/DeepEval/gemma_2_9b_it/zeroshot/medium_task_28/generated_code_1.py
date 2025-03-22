from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Attention Layer
    model.add(layers.Conv2D(filters=1, kernel_size=1, activation='softmax', input_shape=(32, 32, 3))) 
    model.add(layers.Multiply() ) # Multiply attention weights with input

    # Dimensionality Reduction
    model.add(layers.Conv2D(filters=int(32 * 1/3), kernel_size=1))
    model.add(layers.LayerNormalization())
    model.add(layers.ReLU())

    # Dimensionality Restoration
    model.add(layers.Conv2D(filters=32, kernel_size=1))

    # Add to Original Input
    model.add(layers.Add() )

    # Classification Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax')) 

    return model
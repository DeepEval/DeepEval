from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(28, 28, 1))) 

    # 1x1 Conv Layer (Dimensionality Increase)
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))

    # 3x3 Depthwise Separable Conv Layer (Feature Extraction)
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu'))

    # 1x1 Conv Layer (Dimensionality Reduction)
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))

    # Residual Connection
    model.add(layers.Add() )

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(10, activation='softmax')) 

    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Convolutional Base 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  
    model.add(layers.AveragePooling2D((1, 1), strides=(1, 1))) 
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D((4, 4), strides=(4, 4)))

    # Flatten and Concatenate
    model.add(layers.Flatten())
    x = model.output
    x = layers.Flatten()(x) # Flatten each pooling layer output individually
    x = layers.Concatenate()([x]) # Concatenate the flattened outputs

    # Fully Connected Layers
    model.add(layers.Dense(128, activation='relu')) 
    model.add(layers.Dense(10, activation='softmax')) 

    return model
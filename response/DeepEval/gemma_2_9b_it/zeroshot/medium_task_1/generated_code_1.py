import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Convolutional Layer 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    # Convolutional Layer 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Max Pooling Layer
    model.add(layers.MaxPooling2D((2, 2)))

    # Feature Addition
    model.add(layers.Add()([model.layers[-1].output, model.layers[0].output]))

    # Flatten Layer
    model.add(layers.Flatten())

    # Fully Connected Layer 1
    model.add(layers.Dense(128, activation='relu'))

    # Fully Connected Layer 2 (Output Layer)
    model.add(layers.Dense(10, activation='softmax')) 

    return model
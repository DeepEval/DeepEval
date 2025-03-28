from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

    # First max pooling layer
    model.add(layers.MaxPooling2D((1, 1), strides=(1, 1))) 

    # Second max pooling layer
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2))) 

    # Third max pooling layer
    model.add(layers.MaxPooling2D((4, 4), strides=(4, 4))) 

    # Flatten the outputs of each pooling layer
    model.add(layers.Flatten())

    # Concatenate the flattened outputs
    model.add(layers.Concatenate())

    # Fully connected layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model
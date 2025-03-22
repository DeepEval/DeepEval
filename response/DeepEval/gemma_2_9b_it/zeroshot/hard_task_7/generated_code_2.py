import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # Initial convolutional layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))) 

    # Block 1
    model.add(layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)))
    model.add(layers.Conv2D(32, kernel_size=(1, 1), activation='relu'))  
    model.add(layers.SeparableConv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(32, kernel_size=(1, 1), activation='relu'))

    model.add(layers.Lambda(lambda x: tf.concat(x, axis=-1)))

    # Block 2
    model.add(layers.Lambda(lambda x: tf.keras.backend.shape(x)))
    model.add(layers.Reshape((None, None, 4, 8))) # Adjust groups and channels_per_group as needed
    model.add(layers.Permute((2, 3, 1, 0)))
    model.add(layers.Reshape((None, None, 32))) 

    # Final layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    model = keras.Sequential()

    # First Block
    model.add(layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), input_shape=(32, 32, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128))
    model.add(layers.Reshape((1, 128)))

    # Second Block
    model.add(layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=3)))
    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu'))
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))  

    return model
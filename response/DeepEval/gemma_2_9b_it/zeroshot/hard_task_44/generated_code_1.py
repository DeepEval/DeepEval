import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # Block 1
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1)))  
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Concatenate(axis=-1))

    # Block 2
    model.add(layers.Lambda(lambda x: tf.split(x, 4, axis=-1)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Concatenate(axis=-1))

    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model
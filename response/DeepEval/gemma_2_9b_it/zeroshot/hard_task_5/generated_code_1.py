import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    model = models.Sequential()

    # Block 1
    model.add(layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3), input_shape=(32, 32, 3)))
    for _ in range(3):
        model.add(layers.Conv2D(filters=3, kernel_size=1, activation='relu'))
    model.add(layers.Concatenate(axis=3))

    # Block 2
    model.add(layers.Lambda(lambda x: tf.keras.backend.shape(x)[1:])) 
    model.add(layers.Reshape((32, 32, 3, 3))) 
    model.add(layers.Permute((1, 2, 4, 3)))
    model.add(layers.Reshape((32, 32, 9)))

    # Block 3
    model.add(layers.DepthwiseConv2D(kernel_size=3, activation='relu'))
    model.add(layers.Conv2D(filters=3, kernel_size=1, activation='relu'))

    # Shortcut Connection
    model.add(layers.Conv2D(filters=9, kernel_size=1, activation='relu', input_shape=(32, 32, 3)))

    model.add(layers.Add()) 

    # Classification
    model.add(layers.Flatten())
    model.add(layers.Dense(units=10, activation='softmax'))

    return model
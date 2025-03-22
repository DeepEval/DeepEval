import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential()

    # Specialized feature extraction block
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Repeat the block
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.Conv2D(128, (1, 1), activation='relu'))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Global average pooling
    model.add(layers.GlobalAveragePooling2D())

    # Flatten and classify
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model
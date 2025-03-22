import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    # Input layer
    model.add(layers.Input(shape=(32, 32, 3)))

    # Splitting the channels
    model.add(layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(
        )
    )
    
    # Different convolutional kernels
    model.add(layers.Conv2D(32, (1, 1), activation='relu', padding='same')) 
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (5, 5), activation='relu', padding='same')) 

    # Concatenate outputs
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=2)))

    # Flatten and fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    return model
import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.Sequential()

    # Initial convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    # Block 1
    model.add(layers.Lambda(lambda x: tf.split(x, 2, axis=-1)))
    model.add(layers.Conv2D(16, (1, 1), activation='relu')) 
    model.add(layers.SeparableConv2D(16, (3, 3), activation='relu'))
    model.add(layers.Conv2D(16, (1, 1), activation='relu'))
    model.add(layers.Lambda(lambda x: tf.concat(x, axis=-1)))

    # Block 2
    model.add(layers.Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 16))))
    model.add(layers.Lambda(lambda x: tf.split(x, 4, axis=-1)))
    model.add(layers.Lambda(lambda x: tf.transpose(x, [0, 1, 3, 2])))
    model.add(layers.Lambda(lambda x: tf.reshape(x, (-1, 28, 28, 16))))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    
    return model
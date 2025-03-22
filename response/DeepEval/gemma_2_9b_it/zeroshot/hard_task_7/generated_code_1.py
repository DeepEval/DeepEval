import tensorflow as tf
from tensorflow import keras

def dl_model():
    model = keras.Sequential()
    
    # Initial Convolutional Layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    
    # Block 1
    model.add(keras.layers.Lambda(lambda x: tf.split(x, 2, axis=-1)))
    model.add(keras.layers.Conv2D(32, (1, 1), activation='relu'))  
    model.add(keras.layers.DepthwiseConv2D(kernel_size=(3, 3), activation='relu')) 
    model.add(keras.layers.Conv2D(32, (1, 1), activation='relu'))  
    model.add(keras.layers.Lambda(lambda x: tf.concat(x, axis=-1)))
    
    # Block 2
    model.add(keras.layers.Lambda(lambda x: tf.shape(x)))
    model.add(keras.layers.Reshape((28, 28, 2, 16)))
    model.add(keras.layers.Permute((2, 3, 1, 4)))
    model.add(keras.layers.Reshape((28, 28, 32)))
    
    # Final Layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    return model
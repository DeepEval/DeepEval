import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()

    model.add(layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), activation='relu', input_shape=(28, 28, 1))) 
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation='relu'))  
    model.add(layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    
    return model
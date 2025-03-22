import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    model = tf.keras.models.Sequential()
    
    # Input Layer (Adjust input_shape according to your data)
    model.add(layers.Input(shape=(32, 32, 3)))  

    # Split the input into three groups
    model.add(layers.Lambda(lambda x: tf.split(x, 3, axis=-1)))

    # Depthwise Separable Convolutional Layers
    model.add(layers.Conv2D(filters=64, kernel_size=(1, 1), 
                          use_bias=False, 
                          padding='same', activation='relu'))
    model.add(layers.DepthwiseConv2D(kernel_size=(3, 3), 
                                      strides=1, 
                                      use_bias=False, 
                                      padding='same', activation='relu'))
    model.add(layers.DepthwiseConv2D(kernel_size=(5, 5), 
                                      strides=1, 
                                      use_bias=False, 
                                      padding='same', activation='relu'))

    # Concatenate the outputs from the three groups
    model.add(layers.Concatenate(axis=-1))

    # Flatten and Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    return model
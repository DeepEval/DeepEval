import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    input_layer = keras.Input(shape=(28, 28, 1))

    # First Block
    avg_pool1 = layers.AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool2 = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pool3 = layers.AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    flatten1 = layers.Flatten()(avg_pool1)
    flatten2 = layers.Flatten()(avg_pool2)
    flatten3 = layers.Flatten()(avg_pool3)

    concat = layers.Concatenate()([flatten1, flatten2, flatten3])

    # Transformation
    dense = layers.Dense(128, activation='relu')(concat)
    reshape = layers.Reshape((4, 128))(dense)

    # Second Block
    def depthwise_conv(input_tensor):
        conv1 = layers.DepthwiseConv2D(kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        conv2 = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_tensor)
        conv3 = layers.DepthwiseConv2D(kernel_size=(5, 5), strides=1, padding='same', activation='relu')(input_tensor)
        conv4 = layers.DepthwiseConv2D(kernel_size=(7, 7), strides=1, padding='same', activation='relu')(input_tensor)
        return [conv1, conv2, conv3, conv4]

    split = layers.Lambda(lambda x: tf.split(x, 4, axis=-1))(reshape)
    conv_groups = [depthwise_conv(group) for group in split]
    conv_outputs = [layers.Flatten()(conv) for conv in conv_groups]
    concat = layers.Concatenate()([output for output in conv_outputs])

    # Final Layer
    dense = layers.Dense(10, activation='softmax')(concat)

    model = keras.Model(inputs=input_layer, outputs=dense)
    return model
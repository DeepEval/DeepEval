import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():     
    # Input layer
    input_layer = keras.Input(shape=(32, 32, 3))

    # Block 1
    def block1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Split the input tensor along the last dimension
        tensors = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Apply the same convolutional operations to each split tensor
        paths = [layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(tensor) for tensor in tensors]
        paths = [layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(path) for path in paths]
        paths = [layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(path) for path in paths]
        
        # Concatenate the paths
        output_tensor = layers.Concatenate()([path1, path2, path3, paths[0], paths[1], paths[2]])
        
        return output_tensor
    
    block1_output = block1(input_layer)
    
    # Transition Convolution
    transition_conv = layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', activation='relu')(block1_output)

    # Block 2
    def block2(input_tensor):
        # Global max pooling
        max_pooled = layers.GlobalMaxPooling2D()(input_tensor)
        
        # Fully connected layer to generate channel-matching weights
        weights = layers.Dense(32, activation='relu')(max_pooled)
        weights = layers.Dense(32, activation='relu')(weights)
        
        # Reshape the weights to match the shape of the input tensor
        weights = layers.Reshape((1, 1, 32))(weights)
        
        # Multiply the weights with the input tensor
        output_tensor = layers.Multiply()([input_tensor, weights])
        
        return output_tensor
    
    block2_output = block2(transition_conv)

    # Branch
    branch = layers.Lambda(lambda x: x)(input_layer)

    # Add the main path and the branch
    output = layers.Add()([block2_output, branch])

    # Fully connected layer for classification
    output = layers.Dense(10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model
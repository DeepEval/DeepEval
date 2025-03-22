import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function creates a deep learning model for image classification using MNIST dataset.
    
    The model consists of three branches, each composed of a same block. This block first elevates the dimension through a 1x1 convolutional layer,
    then extracts features through a 3x3 depthwise separable convolutional, followed by a 1x1 convolutional layer to reduce the dimension.
    Finally, it adds the block's input to form the output. The outputs from the three branches are concatenated, then passed through a flattening layer
    followed by a fully connected layer to generate classification probabilities.
    
    Args:
    None
    
    Returns:
    A deep learning model for image classification
    """
    
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Define the block for the three branches
    def block(x):
        # Elevate the dimension through a 1x1 convolutional layer
        x = layers.Conv2D(64, (1, 1), activation='relu')(x)
        
        # Extract features through a 3x3 depthwise separable convolutional
        x = layers.DepthwiseConv2D((3, 3), activation='relu')(x)
        
        # Reduce the dimension through a 1x1 convolutional layer
        x = layers.Conv2D(64, (1, 1), activation='relu')(x)
        
        # Add the block's input to form the output
        x = layers.Add()([x, layers.Conv2D(64, (1, 1), activation='relu')(x)])
        
        return x
    
    # Define the three branches
    branch1 = layers.Input(shape=input_shape)
    branch2 = layers.Input(shape=input_shape)
    branch3 = layers.Input(shape=input_shape)
    
    # Apply the block to each branch
    branch1 = block(branch1)
    branch2 = block(branch2)
    branch3 = block(branch3)
    
    # Concatenate the outputs from the three branches
    concat = layers.Concatenate()([branch1, branch2, branch3])
    
    # Apply a flattening layer
    flatten = layers.Flatten()(concat)
    
    # Apply a fully connected layer to generate classification probabilities
    output = layers.Dense(10, activation='softmax')(flatten)
    
    # Create the model
    model = keras.Model(inputs=[branch1, branch2, branch3], outputs=output)
    
    return model
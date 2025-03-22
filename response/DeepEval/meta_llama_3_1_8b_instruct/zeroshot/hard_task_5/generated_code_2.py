# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    The model consists of a main path splitted into four blocks: Block 1, Block 2, Block 3, and Block 1 again.
    """
    
    # Define the input shape of the model
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: split the input into three groups and process each group with a 1x1 convolutional layer
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x = [layers.Conv2D(16, 1, activation='relu')(xi) for xi in x]
    x = layers.Concatenate(axis=-1)(x)
    
    # Block 2: shuffle the channels by reshaping, swapping dimensions, and reshaping back
    channels = 16
    groups = 3
    channels_per_group = channels // groups
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, groups, channels_per_group)))(x)
    x = layers.Permute((1, 2, 4, 3))(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, channels)))(x)
    
    # Block 3: apply a 3x3 depthwise separable convolution
    x = layers.DepthwiseConv2D(3, activation='relu')(x)
    
    # Add a branch that connects directly to the input
    x_branch = inputs
    
    # Combine the outputs from the main path and the branch through an addition operation
    x = layers.Add()([x, x_branch])
    
    # Apply a fully connected layer to complete the classification task
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
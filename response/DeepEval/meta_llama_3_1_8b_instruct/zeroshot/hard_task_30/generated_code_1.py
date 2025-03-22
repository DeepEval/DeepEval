# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model comprises two blocks. The first block features a dual-path structure: the main path and a branch path.
    The main path starts with two convolutional layers to increase the feature map width, followed by another convolutional
    to restore the number of channels to match that of the input layer. The branch path directly connects to the input.
    Both paths are then combined through addition to produce the final output.
    
    The second block splits the input into three groups along the channel by encapsulating tf.split within Lambda layer.
    Each group extracts features using depthwise separable convolutional layers with different kernel sizes: 1x1, 3x3, and 5x5.
    The outputs from these three groups are concatenated.
    
    After establishing the input layer, the model processes features through the two blocks and concludes with two fully connected
    layers that generate classification probabilities.
    
    :return: A constructed Keras model
    """
    
    # Define the input layer
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Define the first block with dual-path structure
    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(3, (3, 3), activation='relu')(x)
    
    # Branch path
    branch_x = input_layer
    
    # Combine the main path and branch path through addition
    x = layers.Add()([x, branch_x])
    
    # Define the second block with depthwise separable convolutional layers
    # Split the input into three groups along the channel
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x)
    
    # Extract features using depthwise separable convolutional layers with different kernel sizes
    x1 = layers.SeparableConv2D(32, (1, 1), activation='relu', use_bias=False)(x[0])
    x2 = layers.SeparableConv2D(32, (3, 3), activation='relu', use_bias=False)(x[1])
    x3 = layers.SeparableConv2D(32, (5, 5), activation='relu', use_bias=False)(x[2])
    
    # Concatenate the outputs from the three groups
    x = layers.Concatenate()( [x1, x2, x3] )
    
    # Process features through the two blocks
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Output layer
    output_layer = layers.Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
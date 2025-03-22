# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: Extract deep features
    block1_conv1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    block1_conv2 = layers.DepthwiseConv2D((3, 3), activation='relu')(block1_conv1)
    block1_conv2 = layers.Conv2D(32, (1, 1), activation='relu')(block1_conv2)
    block1_branch = layers.DepthwiseConv2D((3, 3), activation='relu')(block1_conv1)
    block1_branch = layers.Conv2D(32, (1, 1), activation='relu')(block1_branch)
    
    # Concatenate the primary path and branch path along the channel dimension
    block1_concat = layers.Concatenate()([block1_conv2, block1_branch])
    
    # Block 2: Achieve channel shuffling
    x = layers.GlobalAveragePooling2D()(block1_concat)
    x = layers.Reshape((7, 7, 64))(x)  # 64 is the total number of channels from block1_concat
    x = layers.Permute((1, 3, 2))(x)  # Swap the third and fourth dimensions
    x = layers.Reshape((7, 7, 16, 4))(x)  # Reshape to (height, width, groups, channels_per_group)
    x = layers.Permute((1, 2, 4, 3))(x)  # Swap the third and fourth dimensions
    
    # Flatten the features
    x = layers.Flatten()(x)
    
    # Output layer: fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
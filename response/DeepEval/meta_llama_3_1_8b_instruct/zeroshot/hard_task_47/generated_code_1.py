# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Function to construct a deep learning model using Functional APIs of Keras.
    
    The model consists of two main blocks:
    - Block 1: splits the input into three groups along the last dimension using depthwise separable convolutional layers.
    - Block 2: multiple branches for feature extraction and classification result produced through two fully connected layers.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): the constructed deep learning model.
    """

    # Input layer with shape (32, 32, 3)
    inputs = keras.Input(shape=(32, 32, 3))

    # Block 1: Split the input into three groups along the last dimension
    # and utilize depthwise separable convolutional layers for feature extraction
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Define the feature extraction layers for each group
    group1 = layers.DepthwiseConv2D(kernel_size=(1, 1), padding='same')(x[0])
    group1 = layers.BatchNormalization()(group1)
    group1 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(group1)
    group1 = layers.BatchNormalization()(group1)
    group1 = layers.DepthwiseConv2D(kernel_size=(5, 5), padding='same')(group1)
    group1 = layers.BatchNormalization()(group1)
    group1 = layers.GlobalAveragePooling2D()(group1)
    
    group2 = layers.DepthwiseConv2D(kernel_size=(1, 1), padding='same')(x[1])
    group2 = layers.BatchNormalization()(group2)
    group2 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(group2)
    group2 = layers.BatchNormalization()(group2)
    group2 = layers.GlobalAveragePooling2D()(group2)
    
    group3 = layers.DepthwiseConv2D(kernel_size=(1, 1), padding='same')(x[2])
    group3 = layers.BatchNormalization()(group3)
    group3 = layers.DepthwiseConv2D(kernel_size=(5, 5), padding='same')(group3)
    group3 = layers.BatchNormalization()(group3)
    group3 = layers.GlobalAveragePooling2D()(group3)
    
    # Concatenate the outputs from all groups
    output = layers.Concatenate()([group1, group2, group3])

    # Block 2: multiple branches for feature extraction
    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(output)
    branch1 = layers.AveragePooling2D((2, 2))(branch1)
    branch1 = layers.Conv2D(64, (1, 1), activation='relu')(branch1)
    
    # Branch 2: 1x1 convolution and 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(output)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.AveragePooling2D((2, 2))(branch2)
    branch2 = layers.Conv2D(128, (1, 1), activation='relu')(branch2)
    
    # Branch 3: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(output)
    branch3 = layers.Conv2D(64, (1, 7), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(128, (7, 1), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.AveragePooling2D((2, 2))(branch3)
    branch3 = layers.Conv2D(256, (1, 1), activation='relu')(branch3)
    
    # Concatenate the outputs from all branches
    output = layers.Concatenate()([branch1, branch2, branch3])

    # Classification result produced through two fully connected layers
    output = layers.GlobalAveragePooling2D()(output)
    output = layers.Dense(64, activation='relu')(output)
    output = layers.Dense(10, activation='softmax')(output)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=output)

    return model
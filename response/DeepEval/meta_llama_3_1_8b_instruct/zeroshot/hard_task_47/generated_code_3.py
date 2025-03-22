# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Define the input shape for the model
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Define the Lambda layer for splitting the input into three groups
    split_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Define the first block with three groups of depthwise separable convolutional layers
    group1 = layers.DepthwiseConv2D(kernel_size=(1, 1), padding='same')(split_layer[0])
    group1 = layers.BatchNormalization()(group1)
    group1 = layers.Activation('relu')(group1)
    
    group2 = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')(split_layer[1])
    group2 = layers.BatchNormalization()(group2)
    group2 = layers.Activation('relu')(group2)
    
    group3 = layers.DepthwiseConv2D(kernel_size=(5, 5), padding='same')(split_layer[2])
    group3 = layers.BatchNormalization()(group3)
    group3 = layers.Activation('relu')(group3)
    
    # Concatenate the outputs from the three groups
    concat1 = layers.Concatenate()([group1, group2, group3])
    
    # Define the second block with multiple branches for feature extraction
    branch1 = layers.Conv2D(kernel_size=(1, 1), padding='same')(concat1)
    branch1 = layers.BatchNormalization()(branch1)
    branch1 = layers.Activation('relu')(branch1)
    
    branch2 = layers.Conv2D(kernel_size=(3, 3), padding='same')(concat1)
    branch2 = layers.BatchNormalization()(branch2)
    branch2 = layers.Activation('relu')(branch2)
    
    branch3 = layers.Conv2D(kernel_size=(1, 1), padding='same')(concat1)
    branch3 = layers.Conv2D(kernel_size=(1, 7), padding='same')(branch3)
    branch3 = layers.Conv2D(kernel_size=(7, 1), padding='same')(branch3)
    branch3 = layers.Conv2D(kernel_size=(3, 3), padding='same')(branch3)
    branch3 = layers.BatchNormalization()(branch3)
    branch3 = layers.Activation('relu')(branch3)
    
    branch4 = layers.AveragePooling2D(pool_size=(2, 2))(concat1)
    branch4 = layers.Conv2D(kernel_size=(3, 3), padding='same')(branch4)
    branch4 = layers.BatchNormalization()(branch4)
    branch4 = layers.Activation('relu')(branch4)
    
    # Concatenate the outputs from all branches
    concat2 = layers.Concatenate()([branch1, branch2, branch3, branch4])
    
    # Define the fully connected layers for classification
    flatten = layers.Flatten()(concat2)
    dense1 = layers.Dense(128, activation='relu')(flatten)
    outputs = layers.Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Construct the model
model = dl_model()
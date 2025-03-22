# Import necessary packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

# Define the deep learning model
def dl_model():
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define Block 1 with three parallel paths
    block1_input = Input(shape=input_shape)
    
    # Define the first path (average pooling with window size 1x1 and stride 1x1)
    path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(block1_input)
    path1 = layers.Flatten()(path1)
    path1 = layers.Dropout(0.2)(path1)
    
    # Define the second path (average pooling with window size 2x2 and stride 2x2)
    path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(block1_input)
    path2 = layers.Flatten()(path2)
    path2 = layers.Dropout(0.2)(path2)
    
    # Define the third path (average pooling with window size 4x4 and stride 4x4)
    path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(block1_input)
    path3 = layers.Flatten()(path3)
    path3 = layers.Dropout(0.2)(path3)
    
    # Concatenate the three paths
    concatenated = layers.Concatenate()([path1, path2, path3])
    
    # Define Block 2 with multiple branch connections
    block2_input = layers.Reshape((-1, 1))(concatenated)
    
    # Define the first branch (1x1 convolution)
    branch1 = layers.Conv1D(32, 1, activation='relu')(block2_input)
    
    # Define the second branch (1x1 convolution)
    branch2 = layers.Conv1D(32, 1, activation='relu')(block2_input)
    
    # Define the third branch (3x3 convolution)
    branch3 = layers.Conv1D(32, 3, activation='relu')(block2_input)
    
    # Define the fourth branch (1x1 convolution, 3x3 convolution, 3x3 convolution)
    branch4 = layers.Conv1D(32, 1, activation='relu')(block2_input)
    branch4 = layers.Conv1D(32, 3, activation='relu')(branch4)
    branch4 = layers.Conv1D(32, 3, activation='relu')(branch4)
    
    # Define the fifth branch (average pooling, 1x1 convolution)
    branch5 = layers.AveragePooling1D(pool_size=3)(block2_input)
    branch5 = layers.Conv1D(32, 1, activation='relu')(branch5)
    
    # Concatenate the five branches
    concatenated_branches = layers.Concatenate()([branch1, branch2, branch3, branch4, branch5])
    
    # Define the output layer (two fully connected layers)
    output = layers.Dense(64, activation='relu')(concatenated_branches)
    output = layers.Dense(10, activation='softmax')(output)
    
    # Define the model
    model = Model(inputs=block1_input, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()
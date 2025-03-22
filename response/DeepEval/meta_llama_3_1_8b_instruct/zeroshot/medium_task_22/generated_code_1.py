# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import backend as K
import os

# Define the multi-branch convolutional architecture
def multi_branch_conv(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Branch 1: 3x3 convolutions
    branch1 = Conv2D(32, (3, 3), activation='relu')(inputs)
    branch1 = Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D((2, 2))(branch1)
    
    # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
    branch2 = Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D((2, 2))(branch2)
    
    # Branch 3: max pooling
    branch3 = MaxPooling2D((3, 3))(inputs)
    
    # Concatenate the outputs from the branches
    fused_features = Concatenate()([branch1, branch2, branch3])
    
    # Flatten the fused feature maps
    flattened_features = Flatten()(fused_features)
    
    # Fully connected layers for classification
    outputs = Dense(64, activation='relu')(flattened_features)
    outputs = Dense(10, activation='softmax')(outputs)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Define the deep learning model function
def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Define the model using the multi-branch convolutional architecture
    model = multi_branch_conv(input_shape)
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create an instance of the deep learning model
model = dl_model()
model.summary()
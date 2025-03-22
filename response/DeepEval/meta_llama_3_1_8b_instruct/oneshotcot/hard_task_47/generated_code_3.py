import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Block 1: Split input into three groups and apply depthwise separable convolutional layers
    def group_conv(input_tensor):
        group1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        group1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same')(group1[0])
        group1 = layers.BatchNormalization()(group1)
        group1 = layers.Activation('relu')(group1)
        
        group2 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same')(group1[1])
        group2 = layers.BatchNormalization()(group2)
        group2 = layers.Activation('relu')(group2)
        
        group3 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same')(group1[2])
        group3 = layers.BatchNormalization()(group3)
        group3 = layers.Activation('relu')(group3)
        
        output_tensor = layers.Concatenate()([group1[0], group2[1], group3[2]])
        
        return output_tensor
    
    group_output = group_conv(input_layer)
    
    # Block 2: Apply multiple branches for feature extraction
    def branch1(input_tensor):
        branch1 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Activation('relu')(branch1)
        return branch1
    
    def branch2(input_tensor):
        branch2 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Activation('relu')(branch2)
        branch2 = layers.Conv2D(filters=32, kernel_size=(1, 7), padding='same')(branch2)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Activation('relu')(branch2)
        branch2 = layers.Conv2D(filters=32, kernel_size=(7, 1), padding='same')(branch2)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Activation('relu')(branch2)
        branch2 = layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same')(branch2)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Activation('relu')(branch2)
        return branch2
    
    def branch3(input_tensor):
        branch3 = layers.Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        branch3 = layers.BatchNormalization()(branch3)
        branch3 = layers.Activation('relu')(branch3)
        branch3 = layers.AveragePooling2D(pool_size=(2, 2))(branch3)
        return branch3
    
    branch_output = layers.Concatenate()([branch1(group_output), branch2(group_output), branch3(group_output)])
    
    # Flatten the output and apply two fully connected layers
    flatten_layer = layers.Flatten()(branch_output)
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
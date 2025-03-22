# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Deep learning model for image classification using CIFAR-10 dataset.
    
    This model is composed of two blocks:
    1. The first block splits the input into three groups along the last dimension,
       each processing the input with depthwise separable convolutional layers of
       varying kernel sizes (1x1, 3x3, and 5x5).
    2. The second block features multiple branches for feature extraction.
       Each branch processes the input separately using different configurations.
    Finally, after feature extraction in both blocks, the model outputs the result
    via a flattening layer followed by a fully connected layer.
    """
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: Split the input into three groups
    # and apply depthwise separable convolutional layers
    def group_conv(x, kernel_size):
        x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
        group1 = layers.SeparableConv2D(32, kernel_size, activation='relu', padding='same')(x[0])
        group2 = layers.SeparableConv2D(32, kernel_size, activation='relu', padding='same')(x[1])
        group3 = layers.SeparableConv2D(32, kernel_size, activation='relu', padding='same')(x[2])
        return layers.Concatenate()([group1, group2, group3])
    
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = group_conv(x, 1)
    x = group_conv(x, 3)
    x = group_conv(x, 5)
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)
    
    # Block 2: Multiple branches for feature extraction
    def branch1(x):
        x = layers.SeparableConv2D(64, 1, activation='relu', padding='same')(x)
        x = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)
        return x
    
    def branch2(x):
        x = layers.SeparableConv2D(64, 1, activation='relu', padding='same')(x)
        x = layers.MaxPooling2D(2, strides=2, padding='same')(x)
        x = layers.SeparableConv2D(64, 3, activation='relu', padding='same')(x)
        return x
    
    def branch3(x):
        x = layers.MaxPooling2D(2, strides=2, padding='same')(x)
        x = layers.SeparableConv2D(64, 1, activation='relu', padding='same')(x)
        return x
    
    x = layers.Concatenate()([branch1(x), branch2(x), branch3(x)])
    x = layers.MaxPooling2D(2, strides=2, padding='same')(x)
    
    # Flatten and dense layers for output
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
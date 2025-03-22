import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

def create_block(input_tensor):
    # Elevate dimensions with a 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu')(input_tensor)
    # Depthwise separable convolution
    x = layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
    # Reduce dimensions with a 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    # Add the input to the output (residual connection)
    x = layers.add([x, input_tensor])
    return x

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel
    
    # Create three branches with the same block
    branch1 = create_block(inputs)
    branch2 = create_block(inputs)
    branch3 = create_block(inputs)
    
    # Concatenate the outputs of the three branches
    concatenated = layers.concatenate([branch1, branch2, branch3])
    
    # Flatten the output from the branches
    flattened = layers.Flatten()(concatenated)
    
    # Fully connected layer
    dense = layers.Dense(128, activation='relu')(flattened)
    
    # Output layer with softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(dense)  # 10 classes for MNIST
    
    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()
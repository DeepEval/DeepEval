import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Deep learning model for image classification using the CIFAR-10 dataset.
    
    The model features a multi-branch convolutional architecture that processes input features at various scales using different convolutional kernel sizes and pooling.
    Specifically, it includes three branches: one with 3x3 convolutions, another with 1x1 convolutions followed by two 3x3 convolutions, and a third employing max pooling.
    The outputs from these branches are concatenated to create a multi-scale feature fusion block.
    These fused feature maps are then flattened into a one-dimensional vector and passed through two fully connected layers for classification.
    
    Returns:
    model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the inputs for the model
    inputs = keras.Input(shape=input_shape)

    # Branch 1: 3x3 convolutions
    branch1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Conv2D(64, 3, activation='relu', padding='same')(branch1)
    branch1 = layers.MaxPooling2D((2, 2))(branch1)
    branch1 = layers.Flatten()(branch1)

    # Branch 2: 1x1 convolutions followed by two 3x3 convolutions
    branch2 = layers.Conv2D(64, 1, activation='relu', padding='same')(inputs)
    branch2 = layers.Conv2D(64, 3, activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(128, 3, activation='relu', padding='same')(branch2)
    branch2 = layers.MaxPooling2D((2, 2))(branch2)
    branch2 = layers.Flatten()(branch2)

    # Branch 3: max pooling
    branch3 = layers.MaxPooling2D((3, 3))(inputs)
    branch3 = layers.Conv2D(64, 3, activation='relu', padding='same')(branch3)
    branch3 = layers.MaxPooling2D((2, 2))(branch3)
    branch3 = layers.Flatten()(branch3)

    # Concatenate the outputs from the three branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Dense layers for classification
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
model = dl_model()
model.summary()
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import numpy as np

def dl_model():
    """
    This function generates a deep learning model for image classification using 
    the CIFAR-10 dataset with a multi-branch convolutional structure.

    Returns:
        model (Model): The constructed Keras model.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the paths for the multi-branch convolutional structure
    # Path 1: Single 1x1 convolution
    path1_input = Input(shape=input_shape)
    x1 = layers.Conv2D(64, (1, 1), activation='relu')(path1_input)

    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2_input = Input(shape=input_shape)
    x2 = layers.Conv2D(64, (1, 1), activation='relu')(path2_input)
    x2 = layers.Conv2D(64, (7, 1), activation='relu')(x2)
    x2 = layers.Conv2D(64, (1, 7), activation='relu')(x2)

    # Path 3: 1x1 convolution followed by a combination of two sets of 1x7 and 7x1 convolutions
    path3_input = Input(shape=input_shape)
    x3 = layers.Conv2D(64, (1, 1), activation='relu')(path3_input)
    x3 = layers.Conv2D(64, (7, 1), activation='relu')(x3)
    x3 = layers.Conv2D(64, (1, 7), activation='relu')(x3)
    x3 = layers.Conv2D(64, (7, 1), activation='relu')(x3)
    x3 = layers.Conv2D(64, (1, 7), activation='relu')(x3)

    # Path 4: Average pooling followed by a 1x1 convolution
    path4_input = Input(shape=input_shape)
    x4 = layers.GlobalAveragePooling2D()(path4_input)
    x4 = layers.Conv2D(64, (1, 1), activation='relu')(x4)

    # Fuse the outputs of the paths together through concatenation
    x = layers.Concatenate()([x1, x2, x3, x4])

    # Add a fully connected layer for classification
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=[path1_input, path2_input, path3_input, path4_input], outputs=x)

    return model

# Call the function to get the constructed model
model = dl_model()
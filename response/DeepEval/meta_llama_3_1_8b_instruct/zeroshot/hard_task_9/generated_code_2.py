# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model using the Functional API of Keras for image classification.
    
    The model's main path consists of three feature extraction branches:
    1. A branch with a 1x1 convolution.
    2. A branch with a 1x1 convolution followed by a 3x3 convolution.
    3. A branch with a 1x1 convolution followed by two 3x3 convolutions.
    
    The outputs from these three branches are concatenated, followed by a 1x1 convolution to adjust the output dimensions to match the input image's channel size.
    
    Meanwhile, the branch directly connects to the input, and the main path and the branch are fused together through addition.
    
    Finally, the classification result is output through three fully connected layers.
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the inputs
    inputs = keras.Input(shape=input_shape)

    # Feature extraction branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Feature extraction branch 2: 1x1 convolution followed by a 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)

    # Feature extraction branch 3: 1x1 convolution followed by two 3x3 convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs from the three branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])

    # Adjust the output dimensions to match the input image's channel size
    adjusted = layers.Conv2D(3, (1, 1), activation='relu')(concatenated)

    # Fuse the main path and the branch together through addition
    added = layers.Add()([inputs, adjusted])

    # Output the classification result through three fully connected layers
    x = layers.GlobalAveragePooling2D()(added)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
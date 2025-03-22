# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.

    The model's main path consists of three feature extraction branches:
    1. A branch with a 1x1 convolution.
    2. A branch with a 1x1 convolution followed by a 3x3 convolution.
    3. A branch with a 1x1 convolution followed by two 3x3 convolutions.

    The outputs from these three branches are concatenated, followed by a 1x1 convolution
    to adjust the output dimensions to match the input image's channel size.

    Meanwhile, the branch directly connects to the input, and the main path and the branch
    are fused together through addition.

    Finally, the classification result is output through three fully connected layers.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape)

    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1 = layers.GlobalAveragePooling2D()(branch1)

    # Branch 2: 1x1 convolution + 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = layers.GlobalAveragePooling2D()(branch2)

    # Branch 3: 1x1 convolution + 3x3 convolution + 3x3 convolution
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.GlobalAveragePooling2D()(branch3)

    # Concatenate the outputs from the three branches
    concatenated = layers.Concatenate()([branch1, branch2, branch3])

    # 1x1 convolution to adjust the output dimensions
    x = layers.Conv2D(3, (1, 1), activation='relu')(concatenated)

    # Direct connection to the input
    x = layers.Add()([x, inputs])

    # Flatten the output
    x = layers.Flatten()(x)

    # Classification result through three fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
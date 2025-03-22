# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    The model features two specialized blocks: The first block consists of three parallel paths,
    each utilizing average pooling layers of different scales: 1x1, 2x2, and 4x4, with corresponding strides.
    The results of these pooling operations are flattened into one-dimensional vectors, which are then concatenated
    to form a combined output. Between Block 1 and Block 2, a fully connected layer is applied, followed by a reshape
    operation to convert the output from block 1 into a 4-dimensional tensor suitable for block 2 processing.
    The second block contains three branches for feature extraction. Each branch processes the input through various
    configurations: <1x1 convolution, 3x3 convolution>, <1x1 convolution,1x7 convolution, 7x1 convolution, and 3x3 convolution>,
    as well as average pooling. The outputs from all branches are concatenated to generate the output.
    Ultimately, the model produces classification results through two fully connected layers.
    """

    # Input layer for MNIST dataset
    input_layer = keras.Input(shape=(28, 28, 1))

    # Block 1: Three parallel paths with average pooling layers of different scales
    x1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    x1 = layers.Flatten()(x1)

    x2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    x2 = layers.Flatten()(x2)

    x3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    x3 = layers.Flatten()(x3)

    # Concatenate the outputs from the three parallel paths
    x = layers.Concatenate()([x1, x2, x3])

    # Fully connected layer between Block 1 and Block 2
    x = layers.Dense(128, activation='relu')(x)

    # Reshape the output to a 4-dimensional tensor
    x = layers.Reshape((2, 2, 128))(x)

    # Block 2: Three branches for feature extraction
    # Branch 1: 1x1 convolution and 3x3 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(x)
    branch1 = layers.Conv2D(32, (3, 3), activation='relu')(branch1)
    branch1 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(branch1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, and 3x3 convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(x)
    branch2 = layers.Conv2D(32, (1, 7), activation='relu')(branch2)
    branch2 = layers.Conv2D(32, (7, 1), activation='relu')(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)
    branch2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(branch2)

    # Branch 3: Average pooling
    branch3 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Concatenate the outputs from all branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Flatten the output
    x = layers.Flatten()(x)

    # Two fully connected layers for classification
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()
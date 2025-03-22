import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function defines a deep learning model for image classification using the Functional API of Keras.
    
    The model consists of two special blocks. Block 1 contains three parallel paths, each passing through 
    average pooling layers of three different scales (with pooling windows and strides of 1x1,2x2, and 4x4, 
    respectively). Each pooling result is then flattened into a one-dimensional vector and regularized 
    using a dropout operation. These vectors are concatenated and fused to form the output. Between block 
    1 and block 2, a fully connected layer and reshaping operation are used to transform the output of block 
    1 into a 4-dimensional tensor format suitable for processing by block 2. Block 2 includes multiple branch 
    connections: the input is separately passed through four branches for feature extraction: 
    1x1 convolution, <1x1 convolution, 3x3 convolution>,<1x1 convolution, 3x3 convolution,3x3 convolution>, 
    and <average pooling, 1x1 convolution>. The outputs of these branches are concatenated and fused to form 
    the output. After the above processing, the final classification result is output through two fully connected layers.
    """

    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Block 1: Three parallel paths for feature extraction
    # Path 1: Average pooling with 1x1 window and stride 1
    path1 = layers.AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path1 = layers.Flatten()(path1)
    path1 = layers.Dropout(0.2)(path1)

    # Path 2: Average pooling with 2x2 window and stride 2
    path2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path2 = layers.Flatten()(path2)
    path2 = layers.Dropout(0.2)(path2)

    # Path 3: Average pooling with 4x4 window and stride 4
    path3 = layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    path3 = layers.Flatten()(path3)
    path3 = layers.Dropout(0.2)(path3)

    # Concatenate the outputs of the three paths
    concatenated = layers.Concatenate()([path1, path2, path3])

    # Reshape the concatenated output to 4-dimensional tensor
    reshaped = layers.Reshape((24, 4))(concatenated)

    # Block 2: Multiple branch connections for feature extraction
    # Branch 1: 1x1 convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)

    # Branch 2: <1x1 convolution, 3x3 convolution>
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu')(branch2)

    # Branch 3: <1x1 convolution, 3x3 convolution, 3x3 convolution>
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(reshaped)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu')(branch3)

    # Branch 4: <average pooling, 1x1 convolution>
    branch4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(reshaped)
    branch4 = layers.Conv2D(32, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs of the four branches
    concatenated_branches = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    flattened = layers.Flatten()(concatenated_branches)

    # Two fully connected layers for classification
    x = layers.Dense(128, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
if __name__ == "__main__":
    model = dl_model()
    model.summary()
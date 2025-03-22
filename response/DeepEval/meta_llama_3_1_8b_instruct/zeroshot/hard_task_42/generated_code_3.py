# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model using the Functional API of Keras.
    
    The model consists of two specific blocks. Block 1 contains three parallel paths, each passing through max pooling layers of three different scales.
    The pooling results from each path are then flattened into one-dimensional vectors and regularized using dropout layers. These vectors are concatenated to form the output.
    Between block 1 and block 2, a fully connected layer and reshaping operation are used to transform the output of block 1 into a 4-dimensional tensor format suitable for processing by block 2.
    Block 2 starts with four parallel paths from the same input layer, each employing different convolution and pooling strategies to extract multi-scale features.
    The outputs of all paths are concatenated along the channel dimension. After the above processing, the final classification results are output through two fully connected layers.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model.
    """
    
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Block 1: Three parallel paths with max pooling layers of different scales
    path1 = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(inputs)
    path2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(inputs)
    path3 = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(inputs)
    
    # Concatenate the outputs of the three paths
    output_block1 = layers.Concatenate()([path1, path2, path3])
    
    # Flatten the output of block 1
    output_block1 = layers.Flatten()(output_block1)
    
    # Regularize the output of block 1 using dropout layers
    output_block1 = layers.Dropout(0.2)(output_block1)
    output_block1 = layers.Dropout(0.2)(output_block1)
    
    # Transform the output of block 1 into a 4-dimensional tensor format
    output_block1 = layers.Dense(4 * 4 * 32, activation='relu')(output_block1)
    output_block1 = layers.Reshape((4, 4, 32))(output_block1)
    
    # Block 2: Four parallel paths with different convolution and pooling strategies
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(output_block1)
    
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(output_block1)
    path2 = layers.Conv2D(32, (7, 1), activation='relu')(path2)
    path2 = layers.Conv2D(32, (1, 7), activation='relu')(path2)
    
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(output_block1)
    path3 = layers.Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), activation='relu')(path3)
    path3 = layers.Conv2D(32, (7, 1), activation='relu')(path3)
    path3 = layers.Conv2D(32, (1, 7), activation='relu')(path3)
    
    path4 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(output_block1)
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(path4)
    
    # Concatenate the outputs of the four paths
    output_block2 = layers.Concatenate(axis=-1)([path1, path2, path3, path4])
    
    # Output the final classification results through two fully connected layers
    output_block2 = layers.Flatten()(output_block2)
    outputs = layers.Dense(128, activation='relu')(output_block2)
    outputs = layers.Dense(10, activation='softmax')(outputs)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the model
model = dl_model()
model.summary()
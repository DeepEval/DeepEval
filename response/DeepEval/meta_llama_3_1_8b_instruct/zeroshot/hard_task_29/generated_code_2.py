# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model is structured into two blocks: 
    The first block features both a main path and a branch path. 
    The main path begins with a convolutional layer that increases the feature map's dimensions, 
    followed by another convolutional layer that restores the number of channels. 
    The branch path connects directly to the input. 
    The outputs from both paths are combined through addition to produce the final output of this block.
    
    The second block employs three max pooling layers with varying scales (pooling windows and strides of 1x1, 2x2, and 4x4, respectively). 
    The results from these pooling layers are flattened into one-dimensional vectors, 
    which are then concatenated to form the output for this block.
    
    After processing through both blocks, the model passes through two fully connected layers to produce the final classification result.
    
    Parameters:
    None
    
    Returns:
    A deep learning model for image classification.
    """
    
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)
    
    # Create the input layer
    inputs = layers.Input(shape=input_shape)
    
    # Define the first block
    # Main path
    main_path = layers.Conv2D(32, (3, 3), activation='relu', name='main_conv1')(inputs)
    main_path = layers.Conv2D(16, (3, 3), activation='relu', name='main_conv2')(main_path)
    
    # Branch path
    branch_path = inputs
    
    # Combine the outputs from both paths
    x = layers.Add()([main_path, branch_path])
    
    # Define the second block
    # Apply max pooling with varying scales
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=(1, 1), name='pool1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Concatenate the flattened outputs
    x = layers.Concatenate()([x, layers.GlobalAveragePooling2D()(x)])
    
    # Define the fully connected layers
    x = layers.Dense(128, activation='relu', name='fc1')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the model
model = dl_model()
model.summary()
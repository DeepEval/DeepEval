# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. The main path begins with a 7x7 depthwise convolution 
    to extract features, followed by layer normalization for standardization. Next, it includes two sequential 1x1 
    pointwise convolution layers with the same numbers of channel as the input layer to refine the feature representation. 
    The branch path connects directly to the input. The outputs of both paths are then combined through an addition operation. 
    Finally, the combined output is flattened into a one-dimensional vector and processed through two fully connected layers for classification.
    
    Returns:
        model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the main path
    main_path = keras.Sequential([
        # Apply a 7x7 depthwise convolution to extract features
        layers.DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', input_shape=input_shape),
        # Apply layer normalization for standardization
        layers.LayerNormalization(),
        # Apply two sequential 1x1 pointwise convolution layers to refine the feature representation
        layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', filters=64),
        layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', filters=64),
        # Apply global average pooling to reduce the spatial dimensions
        layers.GlobalAveragePooling2D()
    ])

    # Create the branch path
    branch_path = keras.Sequential([
        # Apply a 1x1 convolution to the input (identity mapping)
        layers.Conv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', filters=64)
    ])

    # Combine the outputs of both paths through an addition operation
    combined_path = keras.Sequential([
        layers.Add()([main_path, branch_path])
    ])

    # Flatten the combined output into a one-dimensional vector
    combined_path = keras.Sequential([
        layers.Flatten()([combined_path])
    ])

    # Apply two fully connected layers for classification
    model = keras.Sequential([
        combined_path,
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model

# Example usage:
model = dl_model()
model.summary()
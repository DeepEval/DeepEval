# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model comprises two specialized blocks:
    Block 1: processes the input through three max pooling layers with varying scales, 
             using pooling windows and strides of 1x1, 2x2, and 4x4, respectively.
             Each pooling result is flattened into a one-dimensional vector and then concatenated.
             Between Block 1 and Block 2, a fully connected layer and a reshape operation 
             convert the output of Block 1 into a 4-dimensional tensor suitable for Block 2.
    Block 2: features multiple branches: the input is processed separately through 
             1x1 convolution, 3x3 convolution, 5x5 convolution, and 3x3 max pooling to extract features.
             The outputs from all branches are then concatenated to form the output.
             Finally, the classification result is generated through a flattening layer 
             followed by a fully connected layer.
    
    Returns:
        A constructed Keras model.
    """

    # Define Block 1
    block_1 = keras.Sequential([
        layers.MaxPooling2D((1, 1), strides=(1, 1)),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),
        layers.MaxPooling2D((4, 4), strides=(4, 4)),
        layers.Flatten(),
        layers.Concatenate()
    ])

    # Define Block 2
    block_2 = keras.Sequential([
        # Branch 1: 1x1 convolution
        keras.Sequential([
            layers.Conv2D(32, (1, 1), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        
        # Branch 2: 3x3 convolution
        keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        
        # Branch 3: 5x5 convolution
        keras.Sequential([
            layers.Conv2D(32, (5, 5), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        
        # Branch 4: 3x3 max pooling
        keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2), strides=(2, 2))
        ]),
        
        # Concatenate outputs from all branches
        layers.Concatenate(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Define the input layer
    input_layer = keras.Input(shape=(28, 28, 1))

    # Convert the input into a 4-dimensional tensor suitable for Block 2
    x = layers.Reshape((7, 7, 4))(block_1(input_layer))

    # Process the input through Block 2
    x = block_2(x)

    # Define the model
    model = keras.Model(input_layer, x)

    return model
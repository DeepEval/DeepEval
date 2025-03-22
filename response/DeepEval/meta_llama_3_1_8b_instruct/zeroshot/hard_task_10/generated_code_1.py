from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model has two feature extraction paths:
    1. One path employs a 1x1 convolution.
    2. The second path consists of a sequence of convolutions: 1x1, followed by 1x7, and then 7x1.
    
    The outputs from these two paths are concatenated, and a 1x1 convolution is applied to align the output dimensions with the input image's channel, creating the output for the main path.
    
    Additionally, a branch connects directly to the input, merging the outputs of the main path and the branch through addition.
    
    Finally, the classification results are produced through two fully connected layers.
    """

    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Path 1: 1x1 convolution
    x = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Path 2: sequence of convolutions (1x1, 1x7, 7x1)
    y = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(64, (1, 7), activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2D(64, (7, 1), activation='relu')(y)
    y = layers.BatchNormalization()(y)
    
    # Concatenate the outputs of the two paths
    x = layers.Concatenate()([x, y])
    
    # Apply a 1x1 convolution to align the output dimensions
    x = layers.Conv2D(64, (1, 1))(x)
    x = layers.BatchNormalization()(x)
    
    # Branch that connects directly to the input
    branch = inputs
    
    # Merge the outputs of the main path and the branch through addition
    x = layers.Add()([x, branch])
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Two fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=x)
    
    return model
# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by adjusting the input feature dimensionality to 16 using a convolutional layer.
    It employs a basic block where the main path includes convolution, batch normalization, and ReLU activation,
    while the branch connects directly to the block's input. The outputs from both paths are combined through an addition operation.
    The core architecture of the model utilizes these basic blocks to create a three-level residual connection structure.
    
    Parameters:
    None
    
    Returns:
    model: A Keras model object representing the constructed deep learning model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model input
    inputs = keras.Input(shape=input_shape)

    # Adjust the input feature dimensionality to 16 using a convolutional layer
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)

    # Define a basic block
    def basic_block(x):
        # Main path: convolution, batch normalization, and ReLU activation
        main_path = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        main_path = layers.BatchNormalization()(main_path)
        
        # Branch: connect directly to the block's input
        branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        branch = layers.BatchNormalization()(branch)
        
        # Combine the outputs from both paths
        return layers.Add()([main_path, branch])

    # First level: a basic block
    x = basic_block(x)

    # Second level: two residual blocks with independent convolutional layers in the branch
    x = basic_block(x)
    x = basic_block(x)

    # Third level: capture features from the initial convolution output using a global branch
    global_branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    global_branch = layers.BatchNormalization()(global_branch)
    x = layers.Add()([x, global_branch])

    # Average pooling followed by a fully connected layer for multi-class classification
    x = layers.AveragePooling2D(pool_size=(8, 8))(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
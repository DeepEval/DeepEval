# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two main components: the main path and the branch path.
    In the main path, the model sequentially extracts features through three 3x3 convolutional layers, followed by a max pooling layer.
    The second path includes a global average pooling layer and two fully connected layers that generate channel weights.
    These weights are reshaped and multiplied with the input.
    Finally, the outputs from both paths are added and passed through two additional fully connected layers for classification.
    
    Returns:
        The constructed model.
    """
    
    # Define the main path
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Sequentially extract features through three 3x3 convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    
    # Max pooling layer
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Main path output
    main_output = x
    
    # Define the branch path
    branch_input = main_output
    
    # Global average pooling layer
    branch_output = layers.GlobalAveragePooling2D()(branch_input)
    
    # Two fully connected layers to generate channel weights
    branch_output = layers.Dense(128, activation='relu')(branch_output)
    branch_output = layers.Dense(32, activation='relu')(branch_output)
    
    # Reshape and multiply with the input
    branch_output = layers.Reshape((1, 1, 32))(branch_output)
    branch_output = layers.Concatenate()([branch_input, branch_output])
    
    # Add the main path output and branch path output
    merged_output = layers.Add()([main_output, branch_output])
    
    # Two additional fully connected layers for classification
    merged_output = layers.Flatten()(merged_output)
    outputs = layers.Dense(10, activation='softmax')(merged_output)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
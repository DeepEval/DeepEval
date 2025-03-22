# Import necessary packages from Keras
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Function to create a deep learning model for image classification.
    
    The model consists of a main path and a branch path. In the main path, 
    the input first undergoes a 1x1 convolutional for dimensionality reduction, 
    followed by two parallel convolutional layers with 1x1 and 3x3 kernels for feature extraction.
    The outputs of these two layers are then concatenated. In the branch path, 
    the input is processed by a 3x3 convolutional layer to ensure that its output dimensions match the main path's channel.
    The outputs from the main and branch paths are combined using addition. 
    Finally, the model produces classification probabilities through a flattening layer followed by two fully connected layers.
    
    Returns:
    model: The constructed deep learning model for image classification.
    """
    
    # Define the input shape
    input_shape = (32, 32, 64)
    
    # Create the main path
    main_path = keras.Sequential([
        layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),  # Dimensionality reduction
        layers.Concatenate([
            layers.Conv2D(32, (1, 1), activation='relu', input_shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
        ])  # Parallel convolutional layers for feature extraction
    ])
    
    # Create the branch path
    branch_path = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)
    ])
    
    # Create the model by combining the main and branch paths
    model = keras.Sequential([
        layers.Add()([main_path, branch_path]),  # Combine the outputs from the main and branch paths
        layers.Flatten(),  # Flattening layer
        layers.Dense(128, activation='relu'),  # First fully connected layer
        layers.Dense(10, activation='softmax')  # Second fully connected layer for classification probabilities
    ])
    
    return model
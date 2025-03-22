# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model comprises two sequential blocks. The first block extracts features through two 3x3 convolutional layers,
    followed by an average pooling layer for smoothing. The input to the first block is combined with the output of the 
    main path via addition. In the second block, the main path compresses the feature map using global average pooling 
    to generate channel weights, which are then refined through two fully connected layers with the same number of 
    channels as the output of the first block. After reshaping, these weights are multiplied by the input. Finally, 
    the flattened output is passed through another fully connected layer for classification.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model
    """
    
    # Define the first block
    first_block = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.AveragePooling2D((2, 2)),
    ])
    
    # Define the main path
    main_path = keras.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.AveragePooling2D((2, 2)),
    ])
    
    # Combine the input with the output of the main path
    combined = keras.Sequential([
        layers.Add()([first_block.input, main_path.output])
    ])
    
    # Define the second block
    second_block = keras.Sequential([
        layers.GlobalAveragePooling2D(),
        layers.Reshape((64, 1)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Reshape((64, 1)),
        layers.Multiply()([combined.output, second_block.output])
    ])
    
    # Flatten the output and pass it through a classification layer
    classification = layers.Flatten()(second_block.output)
    classification = layers.Dense(10, activation='softmax')(classification)
    
    # Define the model
    model = keras.Model(inputs=combined.input, outputs=classification)
    
    return model

# Example usage:
model = dl_model()
model.summary()
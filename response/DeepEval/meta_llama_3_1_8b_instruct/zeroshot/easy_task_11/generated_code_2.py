# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model consists of:
    1. Average pooling layer with a 5x5 window and a 3x3 stride for feature dimensionality reduction.
    2. 1x1 convolutional layer to enhance the depth and introduce nonlinearity to the feature maps.
    3. Flatten layer to transform the feature maps into a 1D format.
    4. Two fully connected layers for additional feature processing with a dropout layer in between.
    5. Output from the second fully connected layer produces a probability distribution across 10 classes.
    
    Parameters:
    None
    
    Returns:
    model: The constructed deep learning model
    """

    # Construct the deep learning model
    model = keras.Sequential([
        # Average pooling layer with a 5x5 window and a 3x3 stride
        layers.AveragePooling2D(pool_size=(5, 5), strides=(3, 3), input_shape=(28, 28, 1)),
        
        # 1x1 convolutional layer to enhance the depth and introduce nonlinearity
        layers.Conv2D(32, (1, 1), activation='relu'),
        
        # Flatten layer to transform the feature maps into a 1D format
        layers.Flatten(),
        
        # First fully connected layer for additional feature processing
        layers.Dense(128, activation='relu'),
        
        # Dropout layer to mitigate overfitting
        layers.Dropout(0.2),
        
        # Second fully connected layer for additional feature processing
        layers.Dense(64, activation='relu'),
        
        # Output layer to produce a probability distribution across 10 classes
        layers.Dense(10, activation='softmax')
    ])
    
    return model

# Usage example:
dl_model_instance = dl_model()
dl_model_instance.summary()
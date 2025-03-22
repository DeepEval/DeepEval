# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

# Define the deep learning model
def dl_model():
    """
    Creates a deep learning model for image classification using the MNIST dataset.
    
    The model consists of an average pooling layer for feature dimensionality reduction,
    followed by a 1x1 convolutional layer for feature enhancement and nonlinearity,
    two fully connected layers for additional feature processing, and a dropout layer to mitigate overfitting.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model.
    """

    # Create the model
    model = keras.Sequential()

    # Add an average pooling layer with a 5x5 window and a 3x3 stride
    model.add(layers.AveragePooling2D(
        pool_size=(5, 5),
        strides=(3, 3),
        input_shape=(28, 28, 1)
    ))

    # Add a 1x1 convolutional layer for feature enhancement and nonlinearity
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation='relu'
    ))

    # Flatten the feature maps
    model.add(layers.Flatten())

    # Add two fully connected layers for additional feature processing
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))

    # Add a dropout layer to mitigate overfitting
    model.add(layers.Dropout(0.2))

    # Add the output layer with a softmax activation function for multi-class classification
    model.add(layers.Dense(10, activation='softmax'))

    return model
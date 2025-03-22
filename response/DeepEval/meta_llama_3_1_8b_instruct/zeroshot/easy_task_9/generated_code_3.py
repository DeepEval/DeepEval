# Import necessary packages from Keras
from tensorflow import keras
from tensorflow.keras import layers

# Define the dl_model function to create a deep learning model for image classification
def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model architecture consists of the following layers:
    1. 1x1 convolutional layer to increase dimensionality
    2. 3x3 depthwise separable convolutional layer for feature extraction
    3. 1x1 convolutional layer to reduce dimensionality
    4. Add the output from the previous layer to the original input layer
    5. Flattening layer to prepare the output for the fully connected layer
    6. Fully connected layer to generate the final classification probabilities
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model
    """

    # Define the input layer with 28x28x1 (grayscale) images
    inputs = layers.Input(shape=(28, 28, 1))

    # Apply a 1x1 convolutional layer to increase dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Apply a 3x3 depthwise separable convolutional layer for feature extraction
    x = layers.DepthwiseConv2D((3, 3), activation='relu')(x)

    # Apply another 1x1 convolutional layer to reduce dimensionality
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)

    # Add the output from the previous layer to the original input layer
    x = layers.Add()([x, inputs])

    # Apply a flattening layer to prepare the output for the fully connected layer
    x = layers.Flatten()(x)

    # Apply a fully connected layer to generate the final classification probabilities
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the deep learning model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
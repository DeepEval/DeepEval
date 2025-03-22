# Import necessary packages
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function defines a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two sequential blocks, each comprising a convolutional layer followed by a max pooling layer,
    which extracts deep features from the input image. The last convolution output dimension needs to be the same as the channel dimension of the input image.
    
    The outputs from both paths are then combined with the input through an addition operation, creating a comprehensive feature representation.
    
    Finally, a fully connected layer classifies the output, providing a probability distribution across ten categories.
    
    Returns:
        A compiled Keras model for image classification.
    """

    # Define the input layer
    inputs = keras.Input(shape=(28, 28, 1), name="input_layer")
    
    # First sequential block
    x = layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1))(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second sequential block
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Add the input to the feature representations
    x = layers.Add()([x, inputs])
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Dense layer for classification
    outputs = layers.Dense(10, activation="softmax")(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs, name="image_classification_model")
    
    # Compile the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    return model

# Example usage:
model = dl_model()
model.summary()
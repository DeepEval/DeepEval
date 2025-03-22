# Import necessary libraries
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define the dl_model function
def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model starts with two sequential layers comprising a convolutional layer followed by a max pooling layer, 
    followed by an additional convolutional layer. After these layers, the feature maps are flattened into a one-dimensional vector.
    This vector is then processed by two fully connected layers to produce the final classification results.
    
    Returns:
    model: The constructed deep learning model.
    """
    
    # Initialize the model using the Functional API of Keras
    inputs = keras.Input(shape=(28, 28, 1), name='img')
    
    # First convolutional layer with max pooling
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Second convolutional layer with max pooling
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the feature maps into a one-dimensional vector
    x = layers.Flatten()(x)
    
    # First fully connected layer
    x = layers.Dense(64, activation='relu')(x)
    
    # Second fully connected layer with output layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
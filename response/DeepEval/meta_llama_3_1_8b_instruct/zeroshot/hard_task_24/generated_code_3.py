# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
from tensorflow.keras import Input

def dl_model():
    """
    Function to create a deep learning model for image classification using Keras Functional APIs.

    Returns:
    model: The constructed deep learning model.
    """
    
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Branch 1: Extract local features through a 3x3 convolutional layer
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    
    # Branch 2: Sequentially pass through a max pooling layer, a 3x3 convolutional layer, and an upsampling layer
    branch2 = layers.MaxPooling2D((2, 2))(inputs)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.UpSampling2D((2, 2))(branch2)
    
    # Branch 3: Sequentially pass through a max pooling layer, a 3x3 convolutional layer, and an upsampling layer
    branch3 = layers.MaxPooling2D((2, 2))(inputs)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    branch3 = layers.UpSampling2D((2, 2))(branch3)
    
    # Fuse the outputs of all branches through concatenation
    merged = layers.Concatenate()([branch1, branch2, branch3])
    
    # Pass the merged output through another 1x1 convolutional layer
    merged = layers.Conv2D(64, (1, 1), activation='relu')(merged)
    
    # Flatten the output
    merged = layers.Flatten()(merged)
    
    # Pass the output through three fully connected layers
    merged = layers.Dense(64, activation='relu')(merged)
    merged = layers.Dense(64, activation='relu')(merged)
    outputs = layers.Dense(10, activation='softmax')(merged)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()
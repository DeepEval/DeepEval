# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model using the Functional API of Keras.
    The model consists of a main path and a branch path for image classification using the MNIST dataset.
    
    Parameters:
    None
    
    Returns:
    A compiled Keras model.
    """
    
    # Define the input shape for the MNIST dataset
    input_shape = (28, 28, 1)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    
    # Branch path
    x_branch = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    
    # Combine the outputs from both paths
    x = layers.Add()([x, x_branch])
    
    # Downsample the spatial dimensions
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()
print(model.summary())
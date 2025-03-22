# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    This function defines a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two 1x1 convolutional layers, followed by a 3x1 convolutional layer, 
    and a 1x3 convolutional layer. The number of channels will be restored to match the input's 
    channel count using another 1x1 convolutional layer. To mitigate overfitting, each 
    convolutional layer includes a dropout layer. The processed features will be combined with 
    the original input via addition. Finally, the model includes a flattening layer followed by 
    a fully connected layer to produce the final probability distribution.
    
    Returns:
    model (tf.keras.Model): The constructed model.
    """

    # Define the input shape of the MNIST dataset (28x28 grayscale images)
    input_shape = (28, 28, 1)
    
    # Define the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Define the first 1x1 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    
    # Define the second 1x1 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Define the 3x1 convolutional layer with dropout
    x = layers.Conv2D(32, (3, 1), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Define the 1x3 convolutional layer with dropout
    x = layers.Conv2D(32, (1, 3), activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Define the layer to restore the number of channels
    x = layers.Conv2D(1, (1, 1))(x)
    
    # Add the original input to the processed features
    x = layers.Add()([inputs, x])
    
    # Define the flattening layer
    x = layers.Flatten()(x)
    
    # Define the fully connected layer to produce the final probability distribution
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
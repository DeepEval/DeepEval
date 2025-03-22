# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using Keras Functional API.
    
    The model comprises a main path and a branch path. The main path extracts global information from the feature map 
    through global average pooling, which is then processed through two fully connected layers to generate weights.
    These weights are reshaped to match the input layer's shape and multiplied element-wise with the original feature map.
    
    The branch path connects to the input layer via a 3x3 convolution, adjusting the output feature map's channel to match that of the input layer.
    The outputs from both paths are added together. Finally, the combined result is passed through three fully connected layers 
    to produce the final probability distribution for classification.
    
    Parameters:
    None
    
    Returns:
    model (keras.Model): The constructed deep learning model
    """

    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Main path
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    
    # Global average pooling to extract global information from the feature map
    x = layers.GlobalAveragePooling2D()(x)
    
    # Process the pooled output through two fully connected layers to generate weights
    weights = layers.Dense(64, activation='relu')(x)
    weights = layers.Dense(32, activation='relu')(weights)
    
    # Reshape the weights to match the input layer's shape
    weights = layers.Reshape((1, 1, 32))(weights)
    
    # Multiply the reshaped weights with the original feature map element-wise
    x = layers.Multiply()([x, weights])
    
    # Branch path
    y = layers.Conv2D(32, (3, 3), activation='relu')(x)
    y = layers.MaxPooling2D((2, 2))(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    y = layers.MaxPooling2D((2, 2))(y)
    y = layers.Conv2D(64, (3, 3), activation='relu')(y)
    y = layers.GlobalAveragePooling2D()(y)
    y = layers.Dense(64, activation='relu')(y)
    y = layers.Dense(32, activation='relu')(y)
    
    # Add the outputs from both paths
    x = layers.Add()([x, y])
    
    # Pass the combined result through three fully connected layers to produce the final probability distribution for classification
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
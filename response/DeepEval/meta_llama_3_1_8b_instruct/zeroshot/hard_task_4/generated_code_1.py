# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, GlobalAveragePooling2D, Dense, Reshape, Conv2D

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by increasing the dimensionality of the input's channels threefold with a 1x1 convolution
    and extracts initial features using a 3x3 depthwise separable convolution. Then, it computes channel attention
    weights through global average pooling followed by two fully connected layers. These weights are then reshaped
    to match the initial features and multiplied with the initial features to achieve channel attention weighting.
    Finally, a 1x1 convolution reduces the dimensionality, and the output is combined with the initial input. The
    resulting output is passed through a flattening layer and a fully connected layer to complete the classification process.
    
    Args:
        None
    
    Returns:
        A constructed Keras model.
    """

    # Define the input shape of the model
    input_shape = (32, 32, 3)
    
    # Create the base model
    inputs = keras.Input(shape=input_shape)
    
    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    x = Conv2D(3 * 3, kernel_size=(1, 1), activation='relu')(inputs)
    
    # Extract initial features using a 3x3 depthwise separable convolution
    x = SeparableConv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(x)
    
    # Compute channel attention weights through global average pooling followed by two fully connected layers
    attention_weights = GlobalAveragePooling2D()(x)
    attention_weights = Dense(64, activation='relu')(attention_weights)
    attention_weights = Dense(32, activation='relu')(attention_weights)
    
    # Reshape the attention weights to match the initial features
    attention_weights = Reshape((1, 1, 32))(attention_weights)
    
    # Multiply the initial features with the attention weights
    x = x * attention_weights
    
    # Reduce the dimensionality with a 1x1 convolution
    x = Conv2D(3, kernel_size=(1, 1), activation='relu')(x)
    
    # Combine the output with the initial input
    x = layers.Add()([inputs, x])
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Complete the classification process with a fully connected layer
    outputs = Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
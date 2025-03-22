# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    The model splits the input image into three channel groups, applies separable convolutional layers 
    to each group, concatenates the outputs, and finally passes them through three fully connected layers.
    
    Returns:
        model (keras.Model): The constructed deep learning model.
    """
    
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Split the input into three channel groups
    channel_group1, channel_group2, channel_group3 = tf.split(inputs, num_or_size_splits=3, axis=-1)
    
    # Apply separable convolutional layers to each group
    channel_group1 = layers.SeparableConv2D(32, (1, 1), activation='relu')(channel_group1)
    channel_group2 = layers.SeparableConv2D(32, (3, 3), activation='relu')(channel_group2)
    channel_group3 = layers.SeparableConv2D(32, (5, 5), activation='relu')(channel_group3)
    
    # Concatenate the outputs from the three groups
    merged = layers.Concatenate()([channel_group1, channel_group2, channel_group3])
    
    # Apply flattening layer
    merged = layers.Flatten()(merged)
    
    # Apply fully connected layers to produce the final probability outputs
    merged = layers.Dense(64, activation='relu')(merged)
    merged = layers.Dense(64, activation='relu')(merged)
    outputs = layers.Dense(10, activation='softmax')(merged)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the function
model = dl_model()
model.summary()
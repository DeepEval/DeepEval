# Import necessary libraries
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model using Keras Functional API for image classification on the CIFAR-10 dataset.
    
    The model consists of the following layers:
    - A 7x7 depthwise separable convolutional layer for extracting spatial features
    - A layer normalization layer for enhancing training stability
    - Two fully connected layers with the same number of channels as the input layer for channel-wise feature transformation
    - An addition operation to combine the original input with the processed features
    - Two final fully connected layers for classification
    """
    
    # Define the input layer with shape (32, 32, 3) representing 32x32 RGB images
    inputs = keras.Input(shape=(32, 32, 3))
    
    # Extract spatial features with a 7x7 depthwise separable convolutional layer
    x = layers.DepthwiseConv2D(kernel_size=7, padding='same')(inputs)
    
    # Apply layer normalization to enhance training stability
    x = layers.LayerNormalization()(x)
    
    # Flatten the output of the convolutional layer
    x = layers.Flatten()(x)
    
    # Pass through two fully connected layers with the same number of channels as the input layer
    # for channel-wise feature transformation
    x = layers.Dense(3072, activation='relu')(x)  # 3072 is the number of channels in the input layer
    x = layers.Dense(3072, activation='relu')(x)
    
    # Combine the original input with the processed features through an addition operation
    x = layers.Add()([inputs, x])
    
    # Pass through two final fully connected layers for classification
    x = layers.Dense(64, activation='relu')(x)  # 64 is a suitable number for the intermediate layer
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 is the number of categories in CIFAR-10
    
    # Define the model
    model = Model(inputs, outputs)
    
    return model

# Construct the deep learning model
model = dl_model()
model.summary()
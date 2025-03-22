# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model for image classification using the Functional API of Keras.
    
    The model consists of four parallel convolutional paths:
    1. The first path utilizes a 1x1 convolution.
    2. The second path features two 3x3 convolutions stacked after a 1x1 convolution.
    3. The third path includes a single 3x3 convolution following a 1x1 convolution.
    4. The fourth path starts with max pooling followed by a 1x1 convolution.
    
    The outputs from these paths are concatenated, flattened, and passed through a dense layer with 128 units before reaching the final output layer,
    which uses softmax activation to classify the input into one of 10 categories.
    
    Returns:
        A Keras Model instance.
    """
    
    # Define the input shape of the images
    input_shape = (32, 32, 3)
    
    # Create the model
    inputs = keras.Input(shape=input_shape)
    
    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu', name='path1_conv1')(inputs)
    path1 = layers.BatchNormalization(name='path1_bn1')(path1)
    
    # Path 2: 1x1 convolution -> 3x3 convolution -> 3x3 convolution
    path2 = layers.Conv2D(32, (1, 1), activation='relu', name='path2_conv1')(inputs)
    path2 = layers.BatchNormalization(name='path2_bn1')(path2)
    path2 = layers.Conv2D(32, (3, 3), activation='relu', name='path2_conv2')(path2)
    path2 = layers.BatchNormalization(name='path2_bn2')(path2)
    path2 = layers.Conv2D(32, (3, 3), activation='relu', name='path2_conv3')(path2)
    
    # Path 3: 1x1 convolution -> 3x3 convolution
    path3 = layers.Conv2D(32, (1, 1), activation='relu', name='path3_conv1')(inputs)
    path3 = layers.BatchNormalization(name='path3_bn1')(path3)
    path3 = layers.Conv2D(32, (3, 3), activation='relu', name='path3_conv2')(path3)
    
    # Path 4: max pooling -> 1x1 convolution
    path4 = layers.MaxPooling2D((2, 2))(inputs)
    path4 = layers.Conv2D(32, (1, 1), activation='relu', name='path4_conv1')(path4)
    
    # Concatenate the outputs from the four paths
    concatenated = layers.Concatenate()([path1, path2, path3, path4])
    
    # Apply convolutional layers
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv1')(concatenated)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Apply dense layers
    x = layers.Dense(128, activation='relu', name='dense1')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    
    # Apply the final output layer with softmax activation
    outputs = layers.Dense(10, activation='softmax', name='output')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
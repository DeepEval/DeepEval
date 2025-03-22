# Import necessary packages from Keras
from tensorflow.keras.layers import Conv2D, Concatenate, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf

def dl_model():
    """
    This function creates a deep learning model for image classification using Functional API of Keras.
    
    The model architecture is as follows:
    1. Compresses the input channels with a 1x1 convolutional layer.
    2. Expands the features through two parallel convolutional layers, applying 1x1 and 3x3 convolutions, and concatenates the results.
    3. Flattens the output feature map into a one-dimensional vector and passes it through two fully connected layers to produce the classification results.
    
    Args:
    None
    
    Returns:
    A compiled Keras model.
    """
    
    # Define the input shape
    input_shape = (32, 32, 64)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Compress the input channels with a 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu', padding='same')(inputs)
    
    # Expand the features through two parallel convolutional layers
    x1 = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    
    # Concatenate the results of the two parallel convolutional layers
    x = Concatenate()([x1, x2])
    
    # Flatten the output feature map into a one-dimensional vector
    x = Flatten()(x)
    
    # Pass the flattened vector through two fully connected layers
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
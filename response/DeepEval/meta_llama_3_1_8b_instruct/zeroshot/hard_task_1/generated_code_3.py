# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins with an initial convolutional layer that adjusts the number of output channels to match the input image channels.
    It then features two parallel processing paths: Path1: Global average pooling followed by two fully connected layers.
    Path2: Global max pooling followed by two fully connected layers. These paths both extract features whose size is equal to the input's channels.
    The outputs from both paths are added and passed through an activation function to generate channel attention weights matching the input's shape,
    which are then applied to the original features through element-wise multiplication.
    Block 2 extracts spatial features by separately applying average pooling and max pooling. The outputs are concatenated along the channel dimension,
    followed by a 1x1 convolution and a sigmoid activation to normalize the features. These normalized features are then multiplied element-wise with the channel dimension features from Block 1.
    Finally, an additional branch with a 1x1 convolutional layer ensures the output channels align with the input channels. The result is added to the main path and activated.
    The final classification is performed through a fully connected layer.
    
    Returns:
    A Keras model instance.
    """
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = layers.Conv2D(32, kernel_size=3, padding='same')(inputs)
    
    # Block 1: Channel attention
    # Path1: Global average pooling followed by two fully connected layers
    x1 = layers.GlobalAveragePooling2D()(x)
    x1 = layers.Dense(64, activation='relu')(x1)
    x1 = layers.Dense(32, activation='relu')(x1)
    
    # Path2: Global max pooling followed by two fully connected layers
    x2 = layers.GlobalMaxPooling2D()(x)
    x2 = layers.Dense(64, activation='relu')(x2)
    x2 = layers.Dense(32, activation='relu')(x2)
    
    # Add the outputs from both paths and apply an activation function
    x_att = layers.Add()([x1, x2])
    x_att = layers.Activation('sigmoid')(x_att)
    
    # Apply channel attention weights to the original features
    x = layers.Lambda(lambda x: x * x_att)(x)
    
    # Block 2: Spatial features
    # Average pooling and max pooling
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    
    # Concatenate the outputs along the channel dimension
    x = layers.Concatenate()([avg_pool, max_pool])
    
    # Normalize the features
    x = layers.Conv2D(32, kernel_size=1, padding='same')(x)
    x = layers.Activation('sigmoid')(x)
    
    # Multiply the normalized features with the channel dimension features from Block 1
    x = layers.Lambda(lambda x: x * x_att)(x)
    
    # Additional branch with a 1x1 convolutional layer
    x_add = layers.Conv2D(32, kernel_size=1, padding='same')(x)
    
    # Add the output from the additional branch to the main path
    x = layers.Add()([x, x_add])
    
    # Activation function
    x = layers.Activation('relu')(x)
    
    # Flatten the output
    x = layers.Flatten()(x)
    
    # Final classification
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
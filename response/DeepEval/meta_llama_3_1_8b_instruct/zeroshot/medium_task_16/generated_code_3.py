# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model begins by splitting the input into three groups along the channel dimension by encapsulating tf.split within Lambda layer 
    and applying 1x1 convolutions to each group independently. The number of convolutional kernels for each group is set to one-third of the input channels.
    
    After this, each group undergoes downsampling via an average pooling layer with consistent parameters.
    
    The three resulting groups of feature maps are then concatenated along the channel dimension.
    
    Finally, the concatenated feature maps are flattened into a one-dimensional vector and passed through two fully connected layers for classification.
    
    Parameters:
    None
    
    Returns:
    model: The constructed deep learning model.
    """

    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = layers.Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply 1x1 convolutions to each group independently
    x1 = layers.Conv2D(int(3 / 3), (1, 1), activation='relu')(x[0])
    x2 = layers.Conv2D(int(3 / 3), (1, 1), activation='relu')(x[1])
    x3 = layers.Conv2D(int(3 / 3), (1, 1), activation='relu')(x[2])
    
    # Downsample each group via an average pooling layer with consistent parameters
    x1 = layers.AveragePooling2D(pool_size=(2, 2))(x1)
    x2 = layers.AveragePooling2D(pool_size=(2, 2))(x2)
    x3 = layers.AveragePooling2D(pool_size=(2, 2))(x3)
    
    # Concatenate the three resulting groups of feature maps along the channel dimension
    x = layers.Concatenate()([x1, x2, x3])
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    x = layers.Flatten()(x)
    
    # Pass the flattened feature maps through two fully connected layers for classification
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Test the function
model = dl_model()
model.summary()
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Create a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two blocks:
    1. The first block processes the input through three max pooling layers with different scales.
    2. The second block splits the input into four groups along the last dimension and processes each group using separable convolutional layers with varying kernel sizes.
    
    Args: None
    Returns:
        model (keras.Model): The constructed deep learning model.
    """

    # Define the input shape of the model
    input_shape = (32, 32, 3)
    
    # Create the input layer
    inputs = keras.Input(shape=input_shape)
    
    # Define the first block
    x = layers.MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(inputs)
    x = layers.Flatten()(x)
    
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(x)
    x = layers.Flatten()(x)
    
    # Apply dropout to mitigate overfitting
    x = layers.Dropout(0.2)(x)
    
    # Transform the output into a four-dimensional tensor for processing in the second block
    x = layers.Reshape((1, -1))(x)
    x = layers.Dense(128, activation='relu')(x)
    
    # Define the second block
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(x)
    
    # Process each group using separable convolutional layers with varying kernel sizes
    x = layers.SeparableConv1D(32, kernel_size=1, activation='relu', padding='same')(x[0])
    x = layers.SeparableConv1D(32, kernel_size=3, activation='relu', padding='same')(x[1])
    x = layers.SeparableConv1D(32, kernel_size=5, activation='relu', padding='same')(x[2])
    x = layers.SeparableConv1D(32, kernel_size=7, activation='relu', padding='same')(x[3])
    
    # Concatenate the outputs from the four groups
    x = layers.Concatenate()(x)
    
    # Flatten the output and pass it through a fully connected layer to produce the classification result
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model
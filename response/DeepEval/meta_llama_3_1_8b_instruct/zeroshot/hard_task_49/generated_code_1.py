# Import necessary packages
from tensorflow.keras import layers, models
from tensorflow.keras import regularizers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model is structured into two blocks:
    1. The first block processes the input through three average pooling layers with varying scales.
    2. The second block splits the input into four groups and processes each group with depthwise separable convolutional layers.
    
    Parameters:
    None
    
    Returns:
    model: The constructed deep learning model.
    """

    # Define the input layer with shape (28, 28, 1) for MNIST dataset
    inputs = layers.Input(shape=(28, 28, 1))

    # First block: three average pooling layers with varying scales
    x = layers.AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(inputs)  # 28x28x1
    x = layers.Flatten()(x)  # 784
    x = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)  # 784
    x = layers.Flatten()(x)  # 196
    x = layers.AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(x)  # 196
    x = layers.Flatten()(x)  # 49

    # Concatenate the flattened vectors
    x = layers.Concatenate()([x, layers.Flatten()(x), layers.Flatten()(x), layers.Flatten()(x)])

    # Transform the output into a 4-dimensional tensor
    x = layers.Reshape((4, 49))(x)

    # Second block: depthwise separable convolutional layers
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(x)
    x1 = layers.SeparableConv2D(32, (1, 1), activation='relu', padding='same')(x[0])  # 1x49x32
    x2 = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x[1])  # 3x49x32
    x3 = layers.SeparableConv2D(32, (5, 5), activation='relu', padding='same')(x[2])  # 5x49x32
    x4 = layers.SeparableConv2D(32, (7, 7), activation='relu', padding='same')(x[3])  # 7x49x32

    # Concatenate the processed groups
    x = layers.Concatenate()([x1, x2, x3, x4])

    # Flatten the output
    x = layers.Flatten()(x)

    # Output layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model
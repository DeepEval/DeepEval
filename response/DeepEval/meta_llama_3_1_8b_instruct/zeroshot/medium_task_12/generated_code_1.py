# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of three blocks, each containing convolutional operations, batch normalization, and a ReLU activation function.
    The output of each block is concatenated with the output from the previous block along the channel dimension, serving as the input for the next block.
    Finally, the output is flattened and passed through two fully connected layers to generate classification probabilities.
    
    Parameters:
    None
    
    Returns:
    model (tf.keras.Model): The constructed deep learning model
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Block 1
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    block1_output = x

    # Block 2
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, block1_output])
    block2_output = x

    # Block 3
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Concatenate()([x, block2_output])
    block3_output = x

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
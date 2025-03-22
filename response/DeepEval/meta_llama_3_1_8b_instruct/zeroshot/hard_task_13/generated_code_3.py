# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of two main blocks: the first block processes the input through four parallel branches 
    and extracts diverse features from the input. The second block reduces the dimensionality of the feature maps, 
    passes them through fully connected layers, and produces the output probability distribution.

    Args:
        None

    Returns:
        model (tf.keras.Model): The constructed deep learning model.
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = layers.Input(shape=input_shape)

    # First block: process the input through four parallel branches
    branch1 = layers.Conv2D(6, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(6, (3, 3), activation='relu')(inputs)
    branch3 = layers.Conv2D(6, (5, 5), activation='relu')(inputs)
    branch4 = layers.MaxPooling2D((3, 3), strides=2, padding='same')(inputs)

    # Concatenate the outputs of the four branches
    concat_output = layers.Concatenate()([branch1, branch2, branch3, branch4])

    # Second block: reduce the dimensionality of the feature maps using global average pooling
    x = layers.GlobalAveragePooling2D()(concat_output)

    # Pass the pooled features through two fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)

    # Reshape the weights to match the input's shape
    weights = layers.Reshape((x.shape[1], x.shape[2], x.shape[3]))(x)

    # Multiply the weights with the input feature map
    x = layers.Multiply()([concat_output, weights])

    # Process the resulting feature map by a final fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
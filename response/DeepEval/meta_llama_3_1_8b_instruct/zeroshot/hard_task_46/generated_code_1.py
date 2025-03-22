# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.

    The model comprises two main blocks:
    1. The first block splits the input into three groups along the channel, each group using separable convolutional 
       with different kernel sizes (1x1, 3x3, and 5x5) to extract features.
    2. The second block features multiple branches for enhanced feature extraction, including a 3x3 convolution, 
       a series of layers consisting of a 1x1 convolution followed by two 3x3 convolutions, and a max pooling branch.

    After processing through both blocks, the concatenated outputs undergo global average pooling, followed by a fully 
    connected layer that produces the final classification results.
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the model using the Functional API
    inputs = keras.Input(shape=input_shape)

    # First block: Split the input into three groups and apply separable convolutions
    x = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    group1 = layers.SeparableConv2D(32, (1, 1), activation='relu', padding='same')(x[0])
    group2 = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(x[1])
    group3 = layers.SeparableConv2D(32, (5, 5), activation='relu', padding='same')(x[2])
    x = layers.Concatenate()([group1, group2, group3])

    # Second block: Multiple branches for enhanced feature extraction
    # Branch 1: 3x3 convolution
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # Branch 2: Series of layers consisting of a 1x1 convolution followed by two 3x3 convolutions
    branch2 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(x)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch2)
    # Branch 3: Max pooling branch
    branch3 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    branch3 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(branch3)
    # Concatenate the outputs from all branches
    x = layers.Concatenate()([branch1, branch2, branch3])

    # Apply global average pooling and a fully connected layer for final classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=x)

    return model

# Call the function to get the constructed model
model = dl_model()
print(model.summary())
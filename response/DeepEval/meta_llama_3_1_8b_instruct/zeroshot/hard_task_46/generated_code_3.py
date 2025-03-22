# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the Functional API of Keras.
    
    The model comprises two main blocks. In the first block, the input is splitted into three groups along the channel 
    by encapsulating tf.split within Lambda layer, each group uses separable convolutional with different kernel sizes 
    (1x1, 3x3, and 5x5) to extract features. The outputs from these three groups are then concatenated.
    
    In the second block, the input is processed through: 
    1.a 3x3 convolution, 
    2.a series of layers consisting of a 1x1 convolution followed by two 3x3 convolutions, 
    3.a max pooling branch. 
    After feature extraction, the outputs from all branches are concatenated for further integration.
    
    After processing through both blocks, the concatenated outputs undergo global average pooling, followed by a fully 
    connected layer that produces the final classification results.
    """

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the first block
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Define separable convolutions for each group
    conv1 = layers.SeparableConv2D(32, (1, 1), activation='relu', use_bias=False, padding='same')(x[0])
    conv2 = layers.SeparableConv2D(32, (3, 3), activation='relu', use_bias=False, padding='same')(x[1])
    conv3 = layers.SeparableConv2D(32, (5, 5), activation='relu', use_bias=False, padding='same')(x[2])
    
    # Concatenate the outputs from the three groups
    x = layers.Concatenate()([conv1, conv2, conv3])

    # Define the second block
    # Branch 1: 3x3 convolution
    conv3x3 = layers.Conv2D(32, (3, 3), activation='relu', use_bias=False, padding='same')(x)
    
    # Branch 2: Series of layers consisting of a 1x1 convolution followed by two 3x3 convolutions
    x1 = layers.Conv2D(32, (1, 1), activation='relu', use_bias=False, padding='same')(x)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', use_bias=False, padding='same')(x1)
    x3 = layers.Conv2D(32, (3, 3), activation='relu', use_bias=False, padding='same')(x2)
    
    # Branch 3: Max pooling branch
    pool = layers.MaxPooling2D((2, 2))(x)
    
    # Concatenate the outputs from all branches
    x = layers.Concatenate()([conv3x3, x2, x3, pool])

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layer
    outputs = layers.Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Usage
model = dl_model()
model.summary()
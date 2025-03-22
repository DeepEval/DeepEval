# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.regularizers import l2

# Define the deep learning model function
def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path and a branch path. The main path begins with a 7x7 depthwise convolution 
    to extract features, followed by layer normalization for standardization. Next, it includes two sequential 
    1x1 pointwise convolution layers with the same numbers of channel as the input layer to refine the feature 
    representation. The branch path connects directly to the input. The outputs of both paths are then combined 
    through an addition operation. Finally, the combined output is flattened into a one-dimensional vector and 
    processed through two fully connected layers for classification.
    
    Args:
        None
    
    Returns:
        A constructed Keras model.
    """

    # Define the input shape and number of classes
    input_shape = (32, 32, 3)
    num_classes = 10

    # Construct the input layer
    inputs = keras.Input(shape=input_shape)

    # Main path
    x = layers.DepthwiseConv2D(32, (7, 7), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Branch path
    branch_x = layers.Conv2D(32, (1, 1), padding='same', use_bias=False)(inputs)
    branch_x = layers.BatchNormalization()(branch_x)
    branch_x = layers.Activation('relu')(branch_x)

    # Combine the main and branch paths
    x = layers.Add()([x, branch_x])

    # Flatten the output
    x = layers.Flatten()(x)

    # Dense layers
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = layers.Dropout(0.2)(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
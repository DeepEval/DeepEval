# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

def dl_model():
    """
    This function constructs a deep learning model for image classification using Functional API of Keras.
    
    The model consists of:
    - A 1x1 initial convolutional layer
    - Three separate branches:
        - Branch 1: extracts local features through a 3x3 convolutional layer
        - Branch 2: sequentially passes through a max pooling layer, a 3x3 convolutional layer, and an upsampling layer
        - Branch 3: sequentially passes through a max pooling layer, a 3x3 convolutional layer, and an upsampling layer
    - The outputs of all branches are fused together through concatenation and passed through another 1x1 convolutional layer
    - Finally, the output is passed through three fully connected layers to produce a 10-class classification result
    
    :return: The constructed model
    """

    # Input layer
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial 1x1 convolutional layer
    x = layers.Conv2D(32, (1, 1), activation='relu', name='conv1')(inputs)

    # Branch 1: extracts local features through a 3x3 convolutional layer
    branch1 = layers.Conv2D(32, (3, 3), activation='relu', name='conv2')(x)

    # Branch 2 and Branch 3: sequentially pass through a max pooling layer, a 3x3 convolutional layer, and an upsampling layer
    x = layers.MaxPooling2D((2, 2), strides=2, name='maxpool1')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv3')(x)
    x = layers.UpSampling2D((2, 2), name='upsample1')(x)

    x = layers.MaxPooling2D((2, 2), strides=2, name='maxpool2')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv4')(x)
    x = layers.UpSampling2D((2, 2), name='upsample2')(x)

    # Fuse the outputs of all branches together through concatenation and pass through another 1x1 convolutional layer
    x = layers.Concatenate()([branch1, x])
    x = layers.Conv2D(32, (1, 1), activation='relu', name='conv5')(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layers
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dense(32, activation='relu', name='dense2')(x)
    outputs = layers.Dense(10, activation='softmax', name='output')(x)

    # Define the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def block1(x):
    # Extract features using separable convolutional layers with different kernel sizes (1x1, 3x3, 5x5)
    conv1x1 = layers.Conv2D(64, (1, 1), padding='same')(x)
    conv3x3 = layers.Conv2D(64, (3, 3), padding='same')(x)
    conv5x5 = layers.Conv2D(64, (5, 5), padding='same')(x)

    # Apply batch normalization to enhance model performance
    conv1x1_bn = layers.BatchNormalization()(conv1x1)
    conv3x3_bn = layers.BatchNormalization()(conv3x3)
    conv5x5_bn = layers.BatchNormalization()(conv5x5)

    # Concatenate the outputs of the three groups
    concat = layers.concatenate([conv1x1_bn, conv3x3_bn, conv5x5_bn])

    return concat

def block2(x):
    # Path 1: 1x1 convolution
    path1 = layers.Conv2D(128, (1, 1), padding='same')(x)

    # Path 2: 3x3 average pooling followed by 1x1 convolution
    path2 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path2 = layers.Conv2D(128, (1, 1), padding='same')(path2)

    # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions
    path3 = layers.Conv2D(128, (1, 1), padding='same')(x)
    path3 = layers.Conv2D(128, (1, 3), padding='same')(path3)
    path3 = layers.Conv2D(128, (3, 1), padding='same')(path3)

    # Path 4: 1x1 convolution followed by 3x3 convolution followed by 1x3 and 3x1 convolutions
    path4 = layers.Conv2D(128, (1, 1), padding='same')(x)
    path4 = layers.Conv2D(128, (3, 3), padding='same')(path4)
    path4 = layers.Conv2D(128, (1, 3), padding='same')(path4)
    path4 = layers.Conv2D(128, (3, 1), padding='same')(path4)

    # Concatenate the outputs of the four paths
    concat = layers.concatenate([path1, path2, path3, path4])

    return concat

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1
    block1_output = block1(input_layer)

    # Block 2
    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = layers.Flatten()(block2_output)
    output_layer = layers.Dense(10, activation='softmax')(flatten_layer)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
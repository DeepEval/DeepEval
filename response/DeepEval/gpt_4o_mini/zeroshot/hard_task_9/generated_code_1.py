import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for CIFAR-10 images (32x32 RGB images)
    input_shape = (32, 32, 3)
    inputs = layers.Input(shape=input_shape)

    # Branch 1: 1x1 Convolution
    branch1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Branch 2: 1x1 Convolution followed by 3x3 Convolution
    branch2 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 Convolution followed by two 3x3 Convolutions
    branch3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)

    # Concatenating the outputs from the three branches
    concatenated = layers.concatenate([branch1, branch2, branch3])

    # 1x1 Convolution to adjust output dimensions
    output_conv = layers.Conv2D(3, (1, 1))(concatenated)

    # Direct connection to input
    residual_connection = layers.add([input, output_conv])

    # Flatten and fully connected layers for classification
    x = layers.Flatten()(residual_connection)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create model
    model = models.Model(inputs=inputs, outputs=x)

    return model
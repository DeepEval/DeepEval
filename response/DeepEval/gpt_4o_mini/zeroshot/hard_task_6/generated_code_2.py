import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def Block1(x):
    # Split the input tensor into three groups
    splits = tf.split(x, num_or_size_splits=3, axis=-1)
    # Process each group with a 1x1 convolution and ReLU activation
    processed_splits = [layers.Conv2D(x.shape[-1] // 3, (1, 1), activation='relu')(split) for split in splits]
    # Concatenate the processed outputs
    return layers.Concatenate(axis=-1)(processed_splits)

def Block2(x):
    # Obtain the shape of the features from Block 1
    shape = tf.shape(x)
    # Reshape to (height, width, groups, channels_per_group)
    x_reshaped = tf.reshape(x, (shape[0], shape[1], shape[2], 3, -1 // 3))
    # Permute dimensions to achieve channel shuffling
    x_permuted = tf.transpose(x_reshaped, perm=[0, 1, 3, 2, 4])
    # Reshape back to the original shape
    return tf.reshape(x_permuted, shape)

def Block3(x):
    # Apply 3x3 depthwise separable convolution
    return layers.SeparableConv2D(x.shape[-1], (3, 3), padding='same', activation='relu')(x)

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32x3
    
    # Main path
    x = Block1(inputs)
    x = Block2(x)
    x = Block3(x)
    x = Block1(x)  # Repeat Block 1

    # Branch path
    branch = layers.AveragePooling2D(pool_size=(8, 8))(inputs)  # Reduce the dimensions
    branch = layers.Conv2D(128, (1, 1), activation='relu')(branch)  # Example of a processing layer

    # Concatenate main path and branch path
    x = layers.Concatenate()([x, branch])

    # Fully connected layer for classification
    x = layers.GlobalAveragePooling2D()(x)  # Global average pooling
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Example usage
model = dl_model()
model.summary()  # Print the model summary to verify the architecture
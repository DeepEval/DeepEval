# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path splitted into four blocks: Block 1, Block 2,Block 3, and Block 1 again.
    In Block 1, the input is splitted into three groups by encapsulating tf.split within Lambda layer,
    each processed by a 1x1 convolutional layer that reduces the number of output channels to one-third of the original.
    The output from these three groups are then concatenated along the channel dimension to generate fused features.
    
    Block 2 starts by obtaining the shape of the feature from Block 1, and then reshapes it into three groups,
    with a target shape of (height, width, groups, channels_per_group), where groups=3 and channels_per_group=channels/groups.
    Then it swaps the third and fourth dimensions using permutation operations, before reshaping the feature back to its original shape
    to achieve channel shuffling.
    
    Block 3 processes the output from Block 2 using a 3x3 depthwise separable convolution.
    
    Additionally, there is a branch in the model that connects directly to the input.
    
    The outputs from the main path and the branch are combined through an addition operation,
    after which the final output is passed through a fully connected layer to complete the classification task.
    
    Returns:
        model (keras.Model): The constructed model.
    """

    # Define the input shape and number of classes for CIFAR-10 dataset
    input_shape = (32, 32, 3)
    num_classes = 10

    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Block 1: Split the input into three groups and process each group with a 1x1 convolutional layer
    block1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    block1 = layers.Conv2D(64 // 3, (1, 1), activation='relu')(block1[0])
    block1 = layers.Conv2D(64 // 3, (1, 1), activation='relu')(block1[1])
    block1 = layers.Conv2D(64 // 3, (1, 1), activation='relu')(block1[2])
    block1 = layers.Concatenate(axis=-1)(block1)

    # Block 2: Reshape, swap dimensions, and reshape back to achieve channel shuffling
    block2 = layers.Lambda(lambda x: tf.shape(x))(block1)
    block2 = layers.Reshape((block2[1], block2[2], 3, 64 // 3))(block1)
    block2 = layers.Permute((3, 4, 1, 2))(block2)
    block2 = layers.Reshape((block2[0], block2[1], block2[2] * block2[3]))(block2)

    # Block 3: Process the output from Block 2 using a 3x3 depthwise separable convolution
    block3 = layers.DepthwiseConv2D((3, 3), activation='relu')(block2)
    block3 = layers.Conv2D(64, (1, 1), activation='relu')(block3)

    # Direct connection branch
    branch = layers.Conv2D(64, (1, 1), activation='relu')(inputs)
    branch = layers.Conv2D(64, (3, 3), activation='relu')(branch)
    branch = layers.Conv2D(64, (1, 1))(branch)

    # Combine the main path and the branch through an addition operation
    outputs = layers.Add()([block3, branch])

    # Final fully connected layer for classification
    outputs = layers.GlobalAveragePooling2D()(outputs)
    outputs = layers.Dense(64, activation='relu')(outputs)
    outputs = layers.Dense(num_classes, activation='softmax')(outputs)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
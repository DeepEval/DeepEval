import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    The model consists of a main path splitted into four blocks: Block 1, Block 2, Block 3, and Block 1 again.
    In Block 1, the input is splitted into three groups by encapsulating tf.split within Lambda layer,
    each processed by a 1x1 convolutional layer that reduces the number of output channels to one-third of the original.
    The output from these three groups are then concatenated along the channel dimension to generate fused features.
    Block 2 starts by obtaining the shape of the feature from Block 1, and then reshapes it into three groups,
    with a target shape of (height, width, groups, channels_per_group), where groups=3 and channels_per_group=channels/groups.
    Then it swaps the third and fourth dimensions using permutation operations, before reshaping the feature back to its original shape to achieve channel shuffling.
    In Block 3, the output from Block 2 is processed using a 3x3 depthwise separable convolution.
    Additionally, there is a branch in the model that connects directly to the input.
    The outputs from the main path and the branch are combined through an addition operation,
    after which the final output is passed through a fully connected layer to complete the classification task.
    """

    # Define the input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1: Split the input into three groups and process each group with a 1x1 convolution
    block1 = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    block1_1 = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(block1[0])
    block1_2 = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(block1[1])
    block1_3 = layers.Conv2D(16, (1, 1), activation='relu', padding='same')(block1[2])
    block1_output = layers.Concatenate()([block1_1, block1_2, block1_3])

    # Block 2: Reshape the output from Block 1, swap the dimensions, and reshape back
    block2 = layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3, 16)))(block1_output)
    block2 = layers.Lambda(lambda x: tf.transpose(x, (0, 1, 2, 4, 3)))(block2)
    block2 = layers.Lambda(lambda x: tf.reshape(x, (-1, 32, 32, 3 * 16)))(block2)

    # Block 3: Process the output from Block 2 with a 3x3 depthwise separable convolution
    block3 = layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(block2)

    # Branch: Connect directly to the input
    branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

    # Combine the outputs from the main path and the branch
    combined_output = layers.Add()([block3, branch])

    # Flatten the output and pass it through a fully connected layer
    output_layer = layers.Flatten()(combined_output)
    output_layer = layers.Dense(10, activation='softmax')(output_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
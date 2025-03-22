# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    Creates a deep learning model for image classification using the CIFAR-10 dataset.
    
    The model consists of a main path splitted into four blocks: Block 1, Block 2, Block 3, and Block 4.
    In Block 1, the input is splitted into three groups by encapsulating tf.split within Lambda layer,
    each processed by a 1x1 convolutional layer that reduces the number of output channels to one-third of the original.
    The output from these three groups are then concatenated along the channel dimension to generate fused features.
    Block 2 starts by obtaining the shape of the feature from Block 1, and then reshapes it into three groups,
    with a target shape of (height, width, groups, channels_per_group), where groups=3 and channels_per_group=channels/groups.
    Then it swaps the third and fourth dimensions using permutation operations, before reshaping the feature back to its original shape to achieve channel shuffling.
    Block 3 processes the output from Block 2 using a 3x3 depthwise separable convolution.
    Additionally, there is a branch in the model that connects directly to the input.
    The outputs from the main path and the branch are combined through an addition operation, after which the final output is passed through a fully connected layer to complete the classification task.
    
    Returns:
        model: A Keras model instance.
    """

    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = keras.Input(shape=input_shape, name='input')

    # Block 1: Split input into three groups and process each group using a 1x1 convolutional layer
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    x = [layers.Conv2D(64 // 3, 1, use_bias=False, name='block1_conv1_group_{}'.format(i))(group) for i, group in enumerate(x)]
    x = layers.Concatenate(name='block1_concat')(x)

    # Block 2: Reshape feature to (height, width, groups, channels_per_group) and perform channel shuffling
    group_size = 64 // 3
    shape = (32, 32, 3, group_size)
    x = layers.Lambda(lambda x: tf.reshape(x, shape))(x)
    x = layers.Permute((1, 2, 4, 3))(x)  # Swap the third and fourth dimensions
    x = layers.Reshape((32, 32, 3, group_size))(x)

    # Block 3: Process output from Block 2 using a 3x3 depthwise separable convolution
    x = layers.SeparableConv2D(64, (3, 3), use_bias=False, name='block3_separable_conv')(x)
    x = layers.BatchNormalization(name='block3_batchnorm')(x)

    # Main path branch
    branch = inputs

    # Combine outputs from main path and branch
    x = layers.Add(name='add')([x, branch])

    # Final fully connected layer for classification
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(10, activation='softmax', name='softmax')(x)

    # Create the model instance
    model = keras.Model(inputs, x, name='deep_learning_model')

    return model
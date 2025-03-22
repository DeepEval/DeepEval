# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    This function constructs a deep learning model for image classification using the MNIST dataset.
    
    The model consists of two main blocks. In Block 1, the primary path extracts deep features through a sequence of operations: 
    a 1x1 convolution, a 3x3 depthwise separable convolution, followed by another 1x1 convolution. Meanwhile, the branch path 
    applies a 3x3 depthwise separable convolution and a 1x1 convolution. The features from both paths are concatenated along 
    the channel dimension. Block 2 starts by obtaining the shape of the features from Block 1 and then reshapes it into four groups, 
    with a target shape of (height, width, groups, channels_per_group). Then it swaps the third and fourth dimensions using 
    permutation operations, before reshaping the features back to its original shape to achieve channel shuffling. The final 
    output is fed into a fully connected layer to complete the classification.
    """

    # Input layer with shape (28, 28, 1) representing the MNIST dataset
    inputs = keras.Input(shape=(28, 28, 1))

    # Block 1
    x = layers.Conv2D(6, (1, 1), activation='relu')(inputs)  # 1x1 convolution
    x = layers.DepthwiseConv2D((3, 3), activation='relu')(x)  # 3x3 depthwise separable convolution
    x = layers.Conv2D(6, (1, 1), activation='relu')(x)  # 1x1 convolution

    # Branch path
    branch_x = layers.DepthwiseConv2D((3, 3), activation='relu')(x)
    branch_x = layers.Conv2D(6, (1, 1), activation='relu')(branch_x)

    # Concatenate features from both paths along the channel dimension
    x = layers.Concatenate()([x, branch_x])

    # Block 2
    block2_input_shape = keras.backend.int_shape(x)[1:]
    x = layers.Reshape((-1, block2_input_shape[0] * block2_input_shape[1], block2_input_shape[2]))(x)
    x = layers.Permute((2, 1))(x)  # Swap the third and fourth dimensions
    x = layers.Reshape(block2_input_shape)(x)  # Reshape back to the original shape

    # Channel shuffling
    x = layers.Reshape((block2_input_shape[0] * block2_input_shape[1], block2_input_shape[2]))(x)
    x = layers.Permute((2, 1))(x)
    x = layers.Reshape(block2_input_shape)(x)

    # Flatten the output
    x = layers.Flatten()(x)

    # Fully connected layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

# Test the function
model = dl_model()
model.summary()
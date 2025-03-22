from tensorflow import keras
from tensorflow.keras import layers

def identity_block(input_tensor, filters):
    """Builds an identity block for the ResNet architecture.

    Args:
        input_tensor: Input tensor to the block.
        filters: List of integers specifying the number of filters in the layers.

    Returns:
        Output tensor from the block.
    """

    # Copy input tensor to use as skip connection
    skip_connection = input_tensor

    # Perform operations within the block
    x = layers.Conv2D(filters[0], (1, 1), strides=(1, 1), padding='valid')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[1], (3, 3), strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters[2], (1, 1), strides=(1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)

    # Add skip connection to output
    x = layers.Add()([x, skip_connection])
    x = layers.Activation('relu')(x)

    return x

def global_average_pooling(input_tensor):
    """Performs global average pooling on an input tensor.

    Args:
        input_tensor: Input tensor to be pooled.

    Returns:
        Output tensor after global average pooling.
    """

    return layers.GlobalAveragePooling2D()(input_tensor)

def block_branch(input_tensor, filters):
    """Builds a branch of the model with a specific block.

    Args:
        input_tensor: Input tensor to the branch.
        filters: List of integers specifying the number of filters in the layers.

    Returns:
        Output tensor from the branch.
    """

    # Apply global average pooling
    pooled_output = global_average_pooling(input_tensor)

    # Pass pooled output through fully connected layers
    dense_output = layers.Dense(filters, activation='relu')(pooled_output)
    dense_output = layers.Dense(input_tensor.shape[-1], activation='sigmoid')(dense_output)

    # Reshape weights to match input shape
    dense_output = layers.Reshape((1, 1, input_tensor.shape[-1]))(dense_output)

    # Element-wise multiplication with block input
    weighted_output = layers.Multiply()([dense_output, input_tensor])

    return weighted_output


def dl_model():
    """Constructs the deep learning model."""

    # Input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # Branch 1
    branch_1 = block_branch(input_img, filters=[64, 64, 64])

    # Branch 2
    branch_2 = block_branch(input_img, filters=[64, 64, 64])

    # Concatenate branch outputs
    concat_output = layers.Concatenate()([branch_1, branch_2])

    # Flatten and fully connected layer
    flatten_output = layers.Flatten()(concat_output)
    output = layers.Dense(10, activation='softmax')(flatten_output)

    # Create model
    model = keras.Model(inputs=input_img, outputs=output)

    return model
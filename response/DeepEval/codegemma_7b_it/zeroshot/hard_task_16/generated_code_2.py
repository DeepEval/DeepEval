from tensorflow.keras.layers import Input, Lambda, Conv2D, concatenate, GlobalMaxPooling2D, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def block_1(input_tensor, filters):
    """
    Creates Block 1 of the model.

    Args:
        input_tensor: Input tensor for the block.
        filters: Number of filters for the convolutions.

    Returns:
        Output tensor for the block.
    """

    # Split the input into three groups along the last dimension.
    x1 = Lambda(lambda x: K.tf.split(x, 3, axis=-1))(input_tensor)

    # Extract deep features from each group.
    x1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='valid')(x1[0])
    x1 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x1)
    x1 = Conv2D(filters=filters, kernel_size=(1, 1), padding='valid')(x1)

    return x1

def transition_conv(input_tensor, filters):
    """
    Creates a transition convolution layer.

    Args:
        input_tensor: Input tensor for the layer.
        filters: Number of filters for the convolution.

    Returns:
        Output tensor for the layer.
    """

    x = Conv2D(filters=filters, kernel_size=(1, 1), padding='valid')(input_tensor)
    x = concatenate([x, input_tensor])

    return x

def block_2(input_tensor, filters, num_classes):
    """
    Creates Block 2 of the model.

    Args:
        input_tensor: Input tensor for the block.
        filters: Number of filters for the convolutions.
        num_classes: Number of classes for classification.

    Returns:
        Output tensor for the block.
    """

    # Global max pooling.
    x = GlobalMaxPooling2D()(input_tensor)

    # Generate channel-matching weights.
    x = Dense(filters)(x)
    x = Dense(filters)(x)

    # Reshape weights to match the shape of the adjusted output.
    x = Reshape((filters, 1, 1))(x)

    # Multiply weights with the adjusted output.
    x = multiply([x, input_tensor])

    # Branch of the model connected directly to the input.
    branch = Conv2D(filters=filters, kernel_size=(1, 1), padding='same')(input_tensor)

    # Add outputs from the main path and the branch.
    output = add([x, branch])

    # Classification layer.
    output = Dense(num_classes, activation='softmax')(output)

    return output

def dl_model():
    """
    Creates the deep learning model.

    Returns:
        The constructed model.
    """

    # Input tensor.
    input_tensor = Input(shape=(32, 32, 3))

    # Block 1.
    x = block_1(input_tensor, filters=64)

    # Transition convolution.
    x = transition_conv(x, filters=64)

    # Block 2.
    output = block_2(x, filters=64, num_classes=10)

    # Create the model.
    model = Model(inputs=input_tensor, outputs=output)

    return model
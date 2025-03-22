import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Dense, Flatten, DepthwiseConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    def block1(input_tensor):
        # Split the input tensor into 3 groups
        split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Apply 1x1 convolutions on each group
        conv_groups = [Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(group) for group in split_groups]
        # Concatenate the groups along the channel dimension
        output_tensor = Concatenate(axis=-1)(conv_groups)
        return output_tensor

    def block2(input_tensor):
        # Get input shape
        input_shape = tf.shape(input_tensor)
        channels = input_tensor.shape[-1]
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = tf.reshape(input_tensor, [-1, input_shape[1], input_shape[2], 3, channels // 3])
        # Permute dimensions to achieve channel shuffling
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        # Reshape back to the original shape
        output_tensor = tf.reshape(permuted, [-1, input_shape[1], input_shape[2], channels])
        return output_tensor

    def block3(input_tensor):
        # Apply a 3x3 depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch directly from the input
    branch = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Main path through the blocks
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(x)

    # Combine outputs from the main path and the branch
    combined = Add()([x, branch])

    # Pass through a fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
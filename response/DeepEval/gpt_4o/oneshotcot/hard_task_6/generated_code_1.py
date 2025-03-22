import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dense, Flatten, Lambda, Concatenate, DepthwiseConv2D, Reshape
import tensorflow as tf

def dl_model():

    def block1(input_tensor):
        # Splitting the input into 3 groups
        channels = input_tensor.shape[-1]
        split_groups = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        
        # Apply a 1x1 convolution to each group
        convs = [Conv2D(channels // 3, kernel_size=(1, 1), activation='relu')(group) for group in split_groups]
        
        # Concatenate the outputs
        output_tensor = Concatenate()(convs)
        return output_tensor

    def block2(input_tensor):
        # Obtain the shape and calculate groups
        input_shape = tf.shape(input_tensor)
        height, width, channels = input_shape[1], input_shape[2], input_tensor.shape[-1]
        groups = 3

        # Reshape, permute, and reshape back for channel shuffling
        reshaped = Reshape((height, width, groups, channels // groups))(input_tensor)
        permuted = tf.transpose(reshaped, perm=(0, 1, 2, 4, 3))  # Swap the last two dimensions
        reshaped_back = Reshape((height, width, channels))(permuted)

        return reshaped_back

    def block3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    x = block1(input_layer)
    x = block2(x)
    x = block3(x)
    x = block1(x)  # Repeating Block 1

    # Branch path
    branch = AveragePooling2D(pool_size=(32, 32))(input_layer)
    branch_flat = Flatten()(branch)

    # Concatenate main path and branch path
    main_flat = Flatten()(x)
    concatenated = Concatenate()([main_flat, branch_flat])

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(concatenated)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Flatten, Dense, Add, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Block 1
    def block_1(input_tensor):
        # Splitting input into 3 groups
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        # Processing each group with a 1x1 convolution
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        # Concatenating the outputs
        output_tensor = Concatenate(axis=-1)([conv1, conv2, conv3])
        return output_tensor

    # Block 2
    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        # Reshape to (height, width, groups, channels_per_group)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
        # Permuting the dimensions to achieve channel shuffling
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(reshaped)
        # Reshape back to original shape
        output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[3]))(permuted)
        return output_tensor

    # Block 3
    def block_3(input_tensor):
        # Applying depthwise separable convolution
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise

    # Constructing the model
    block1_output = block_1(input_layer)
    block2_output = block_2(block1_output)
    block3_output = block_3(block2_output)

    # Adding a direct branch from the input to the output
    branch_output = Conv2D(filters=input_layer.shape[-1], kernel_size=(1, 1), padding='same')(input_layer)

    # Combining the outputs from the main path and the branch
    combined_output = Add()([block3_output, branch_output])

    # Final classification
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Concatenate, DepthwiseConv2D, Add, Flatten, Dense, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[0])
        conv2 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[1])
        conv3 = Conv2D(filters=input_tensor.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs_groups[2])
        output_tensor = Concatenate(axis=-1)([conv1, conv2, conv3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Block 2
    def block_2(input_tensor):
        shape = tf.shape(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 3, shape[3] // 3))(input_tensor)
        permuted = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(reshaped)  # Swap groups and channels
        reshaped_back = Reshape(target_shape=shape)(permuted)
        return reshaped_back

    block2_output = block_2(input_tensor=block1_output)

    # Block 3
    def block_3(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        return depthwise_conv

    block3_output = block_3(input_tensor=block2_output)

    # Direct branch connecting to input
    branch_output = input_layer

    # Combine main path and branch
    combined_output = Add()([block3_output, branch_output])

    # Fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
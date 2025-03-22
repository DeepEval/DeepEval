import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, DepthwiseConv2D, Concatenate, Reshape, Permute, Flatten, Dense


def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):

        input_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_groups[0])
        dwconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dwconv)

        branch_path = input_groups[1]
        output_tensor = Concatenate()([conv2, branch_path]) 

        return output_tensor

    def block_2(input_tensor):

        batchsize, height, width, num_channels = input_tensor.shape
        groups = 4
        channels_per_group = num_channels // groups
        reshape = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        permute = Permute(dims=(1, 2, 4, 3))(reshape)
        output_tensor = Reshape(target_shape=(height, width, num_channels))(permute)

        return output_tensor

    inital_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block_1_output = block_1(input_tensor=inital_conv)
    block_2_output = block_2(input_tensor=block_1_output)

    flatten = Flatten()(block_2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model


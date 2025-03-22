import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Lambda, Concatenate, Reshape, Permute, DepthwiseConv2D, AveragePooling2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):

        input_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        input_channels = input_tensor.shape[-1]
        conv_1 = Conv2D(input_channels // 3, kernel_size=(1, 1), padding='same', activation='relu')(input_groups[0])
        conv_2 = Conv2D(input_channels // 3, kernel_size=(1, 1), padding='same', activation='relu')(input_groups[1])
        conv_3 = Conv2D(input_channels // 3, kernel_size=(1, 1), padding='same', activation='relu')(input_groups[2])
        output_tensor = Concatenate()([conv_1, conv_2, conv_3])

        return output_tensor

    def block_2(input_tensor):

        batchsize, height, width, num_channels = input_tensor.shape
        groups = 3
        channels_per_group = num_channels // groups
        reshape = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        permute = Permute(dims=(1, 2, 4, 3))(reshape)
        output_tensor = Reshape(target_shape=(height, width, num_channels))(permute)

        return output_tensor

    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)
    depthwise_conv_output = DepthwiseConv2D(kernel_size=(3, 3), strides=(2,2), padding='same', activation='relu')(block_2_output)
    main_path = block_1(input_tensor=depthwise_conv_output)

    branch_path = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_layer)
    final_output = Concatenate()([main_path, branch_path])
    output_layer = Dense(units=10, activation='softmax')(final_output)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

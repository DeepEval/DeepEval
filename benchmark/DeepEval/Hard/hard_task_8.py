from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Dense


def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):

        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        dwconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(conv)
        main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dwconv)
        
        conv2_1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2_1)
        
        output_tensor = Concatenate()([main_path, branch_path])

        return output_tensor

    def block_2(input_tensor):

        batchsize, height, width, num_channels = input_tensor.shape
        groups = 4
        channels_per_group = num_channels // groups
        reshape = Reshape(target_shape=(height, width, groups, channels_per_group))(input_tensor)
        permute = Permute(dims=(1, 2, 4, 3))(reshape)
        output_tensor = Reshape(target_shape=(height, width, num_channels))(permute)

        return output_tensor

    block_1_output = block_1(input_tensor=input_layer)
    block_2_output = block_2(input_tensor=block_1_output)
    
    output_layer = Dense(units=10, activation='softmax')(block_2_output)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

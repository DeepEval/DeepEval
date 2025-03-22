import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        branch_path = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
        output_tensor = Concatenate()([conv2, branch_conv])
        return output_tensor

    def block_2(input_tensor):
        shape = keras.backend.int_shape(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 2, shape[3]//2))(input_tensor)
        permuted = Permute((2, 3, 1, 4))(reshaped)
        output_tensor = Reshape(target_shape=(shape[1], shape[2], shape[3]//2))(permuted)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
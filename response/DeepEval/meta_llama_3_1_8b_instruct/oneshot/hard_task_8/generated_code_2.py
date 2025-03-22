import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose, Reshape, Permute, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define Block 1
    def block1(input_tensor):
        depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        
        branch_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_conv)

        output_tensor = Concatenate()([conv2, branch])
        return output_tensor

    block1_output = block1(conv1)

    # Define Block 2
    def block2(input_tensor):
        output_shape = keras.backend.int_shape(input_tensor)
        height, width, channels = output_shape[1], output_shape[2], output_shape[3]
        group_size = channels // 4

        reshape = Reshape((height, width, group_size, 4))(input_tensor)
        permute = Permute((1, 2, 4, 3))(reshape)
        output_tensor = Reshape((height, width, channels))(permute)
        return output_tensor

    block2_output = block2(block1_output)

    dense_layer = Dense(units=10, activation='softmax')(block2_output)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, AveragePooling2D, Reshape, Permute, DepthwiseConv2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):
        # Split the input into three groups
        split_1 = Lambda(lambda x: x[:, :16, :16, :])(input_tensor)
        split_2 = Lambda(lambda x: x[:, 16:, :16, :])(input_tensor)
        split_3 = Lambda(lambda x: x[:, :16, 16:, :])(input_tensor)
        
        # Process each group with a 1x1 convolutional layer
        conv_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_1)
        conv_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_2)
        conv_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split_3)
        
        # Concatenate the outputs
        output_tensor = Concatenate()([conv_1, conv_2, conv_3])
        return output_tensor

    def block2(input_tensor):
        # Get the shape of the input
        shape = input_tensor.get_shape().as_list()
        height, width, channels = shape[1], shape[2], shape[3]
        
        # Reshape and permute to achieve channel shuffling
        reshaped = Reshape((height, width, 3, int(channels / 3)))(input_tensor)
        permuted = Permute((1, 2, 4, 3))(reshaped)
        permuted_shape = permuted.get_shape().as_list()
        
        # Flatten the shape back to original
        output_tensor = Reshape((height, width, channels))(permuted)
        return output_tensor

    def block3(input_tensor):
        # Apply 3x3 depthwise separable convolution
        output_tensor = DepthwiseConv2D(kernel_size=(3, 3))(input_tensor)
        return output_tensor

    # Main path
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)
    block3_output = block3(block2_output)
    block1_repeated_output = block1(block3_output)

    # Branch path
    branch_output = AveragePooling2D(pool_size=(4, 4))(input_layer)
    branch_output = Flatten()(branch_output)

    # Concatenate main path and branch path outputs
    concatenated_output = Concatenate()([block1_repeated_output, branch_output])

    # Fully connected layer
    dense_output = Dense(units=10, activation='softmax')(concatenated_output)

    model = keras.Model(inputs=input_layer, outputs=dense_output)

    return model
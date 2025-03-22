import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, Reshape, Permute, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Primary path
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthconv)
        
        # Branch path
        depthconv_branch = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthconv_branch)
        
        # Concatenate features from both paths
        output_tensor = Concatenate()([conv2, conv_branch])
        return output_tensor

    def block_2(input_tensor):
        # Get shape of the features from Block 1
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        
        # Reshape features into four groups
        reshaped = Reshape(target_shape=(shape[1], shape[2], 4, 64))(input_tensor)
        
        # Swap third and fourth dimensions
        permuted = Permute((3, 4, 1, 2))(reshaped)
        
        # Reshape features back to its original shape
        reshaped_back = Reshape(target_shape=(shape[1], shape[2], 4*64))(permuted)
        
        return reshaped_back

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
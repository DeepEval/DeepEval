import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)

    def block1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthconv)
        output_tensor = Concatenate()([conv1, conv2])
        return output_tensor
    
    def block2(input_tensor):
        # obtain the shape of the input features
        shape = keras.backend.int_shape(input_tensor)
        # reshape the features into four groups
        reshaped = Reshape((shape[1] * shape[2], shape[3]))(input_tensor)
        # swap the third and fourth dimensions
        permuted = Permute((2, 1))(reshaped)
        # reshape the features back to its original shape
        reshaped_back = Reshape((shape[1], shape[2], shape[3] // 4))(permuted)
        return reshaped_back
    
    block1_output = block1(max_pooling)
    block2_output = block2(block1_output)
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
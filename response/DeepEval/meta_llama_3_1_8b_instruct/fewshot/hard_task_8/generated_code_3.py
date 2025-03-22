import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Reshape, Permute, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # primary path
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        # branch path
        conv4 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv4)
        
        # concatenate along channel dimension
        output_tensor = Concatenate()([conv3, conv5])
        return output_tensor
    
    block1_output = block_1(input_layer)
    
    # obtain shape and reshape to groups
    shape_layer = keras.layers.Lambda(lambda x: keras.backend.int_shape(x))(block1_output)
    reshaped = Reshape(target_shape=keras.backend.int_shape(block1_output)[1:])(shape_layer)
    
    # swap third and fourth dimensions
    permuted = Permute([1, 2, 4, 3])(reshaped)
    
    # reshape back to original shape
    reshaped_back = Reshape(target_shape=keras.backend.int_shape(block1_output))(permuted)
    
    # channel shuffling
    output_tensor = reshaped_back
    
    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
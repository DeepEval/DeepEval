import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Lambda, Concatenate, Flatten, Reshape, Permute, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block1(input_tensor):
        split_tensor = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(input_tensor)
        
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        branch1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch1)
        
        branch2 = split_tensor[1] 

        output_tensor = Concatenate()([branch1, branch2])
        return output_tensor
    
    block1_output = block1(conv1)

    def block2(input_tensor):
        shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        reshaped = Reshape(target_shape=(shape[1], shape[2], 4, 8))(input_tensor)
        permuted = Permute(dims=(0, 1, 3, 2))(reshaped)
        reshaped_back = Reshape(target_shape=(shape[1], shape[2], 32))(permuted)
        return reshaped_back

    block2_output = block2(block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
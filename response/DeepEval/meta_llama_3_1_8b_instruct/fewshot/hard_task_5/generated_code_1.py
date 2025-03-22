import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Reshape, Permute, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def block_1(input_tensor):
        split_output = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=int(input_tensor.shape[-1]/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[0])
        conv2 = Conv2D(filters=int(input_tensor.shape[-1]/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[1])
        conv3 = Conv2D(filters=int(input_tensor.shape[-1]/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_output[2])
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor
    
    def block_2(input_tensor):
        feature_shape = Lambda(lambda x: tf.shape(x))(input_tensor)
        reshaped = Reshape(target_shape=(feature_shape[1], feature_shape[2], 3, int(feature_shape[-1]/3)))(input_tensor)
        permuted = Permute()(reshaped)
        output_tensor = Reshape(target_shape=feature_shape)(permuted)
        return output_tensor
    
    def block_3(input_tensor):
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv
    
    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    block3_output = block_3(input_tensor=block2_output)
    block1_output_again = block_1(input_tensor=input_layer)
    
    branch = block1_output_again
    
    adding_layer = Add()([block3_output, branch])
    flatten = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
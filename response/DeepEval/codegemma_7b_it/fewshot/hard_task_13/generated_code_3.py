import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, GlobalAveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool4])
        return output_tensor

    def block_2(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(gap)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        reshaped = Reshape(target_shape=input_tensor.shape)(dense2)
        output_tensor = tf.multiply(reshaped, input_tensor)
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
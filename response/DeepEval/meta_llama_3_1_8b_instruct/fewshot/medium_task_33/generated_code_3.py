import keras
import tensorflow as tf
from keras.layers import Input, DepthwiseConv2D, Concatenate, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        group1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1, num=None, axis=0)[0])(input_tensor)
        group2 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1, num=None, axis=0)[1])(input_tensor)
        group3 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1, num=None, axis=0)[2])(input_tensor)
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)
        output_tensor = Concatenate()([conv1, conv2, conv3])
        return output_tensor

    block_output = block(input_tensor=input_layer)
    flatten = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
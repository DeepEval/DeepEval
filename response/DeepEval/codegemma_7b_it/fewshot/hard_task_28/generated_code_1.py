import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, LayerNormalization

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def main_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2,2), padding='same', activation='relu')(input_tensor)
        l_norm = LayerNormalization()(conv)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(l_norm)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def branch_path(input_tensor):
        return input_tensor

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    output = keras.layers.Add()([main_path_output, branch_path_output])
    flatten = Flatten()(output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
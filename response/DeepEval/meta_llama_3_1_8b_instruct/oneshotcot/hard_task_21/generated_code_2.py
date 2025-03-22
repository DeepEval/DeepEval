import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return path1, path2, path3

    def branch_path(input_tensor):
        aligned_channels = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return aligned_channels

    # main path
    path1, path2, path3 = Lambda(lambda x: K.concatenate([main_path(x)[0], main_path(x)[1], main_path(x)[2]], axis=-1))(input_layer)

    # branch path
    branch_output = branch_path(input_layer)

    # add the main and branch paths
    output_tensor = Lambda(lambda x: K.add(x[0], x[1]))([path1, branch_output])

    # apply batch normalization and flatten
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)

    # add dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
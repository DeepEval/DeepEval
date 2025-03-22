import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(global_avg_pool)
        dense2 = Dense(units=input_tensor.shape[3], activation='sigmoid')(dense1)
        reshaped = Reshape(target_shape=(1, 1, input_tensor.shape[3]))(dense2)
        upsample = UpSampling2D(size=(input_tensor.shape[1], input_tensor.shape[2]))(reshaped)
        output_tensor = Multiply()([input_tensor, upsample])
        return output_tensor

    def branch_path(input_tensor):
        conv = Conv2D(filters=input_tensor.shape[3], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    combined = Add()([main_path_output, branch_path_output])
    flatten = Flatten()(combined)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
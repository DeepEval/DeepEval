import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def path1(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    def path2(input_tensor):
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
        return conv

    def path3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        concat = Concatenate()([conv2_1, conv2_2])
        return concat

    def path4(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        concat = Concatenate()([conv2_1, conv2_2])
        return concat

    path1_output = path1(input_tensor=input_layer)
    path2_output = path2(input_tensor=input_layer)
    path3_output = path3(input_tensor=input_layer)
    path4_output = path4(input_tensor=input_layer)

    concat = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    flatten = Flatten()(concat)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
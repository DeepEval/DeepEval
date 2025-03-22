import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Add, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def feature_branch(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        return conv3

    def fusion_branch(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        return maxpool

    branch1_output = feature_branch(input_tensor=input_layer)
    branch2_output = feature_branch(input_tensor=input_layer)
    branch3_output = feature_branch(input_tensor=input_layer)
    branch4_output = fusion_branch(input_tensor=input_layer)

    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output, branch4_output])
    adjust_dims = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)
    addition = Add()([input_layer, adjust_dims])
    flatten_layer = Flatten()(addition)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
import keras
from keras.layers import Input, Lambda, DepthwiseConv2D, Conv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: tf.split(x, 3, -1))(input_layer)

    group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)

    concat_layer = Concatenate()([conv1x1, conv3x3, conv5x5])

    bath_norm = BatchNormalization()(concat_layer)
    flatten_layer = Flatten()(bath_norm)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
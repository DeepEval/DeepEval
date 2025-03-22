import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, Dropout, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)
        conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_1)
        return conv_1x1_2

    conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_1)
    conv_special = specialized_block(conv_1x1_2)
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_special)
    dropout = Dropout(0.3)(avg_pool)

    for _ in range(2):
        conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout)
        conv_1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1_1)
        conv_special = specialized_block(conv_1x1_2)
        avg_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_special)
        dropout = Dropout(0.3)(avg_pool)

    global_avg_pool = GlobalAveragePooling2D()(dropout)
    flatten = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
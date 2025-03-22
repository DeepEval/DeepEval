import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense, Lambda, Concatenate, Add, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_output = Concatenate()([conv1_1, conv1_2, conv1_3])

    # Branch path
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path_output = Concatenate()([main_path_output, conv2_1])

    # Final model
    flatten = Flatten()(main_path_output)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model
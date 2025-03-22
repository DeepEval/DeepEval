import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K
from keras.backend import tf as tf
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    restore_path = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = input_layer
    block_output = Add()([restore_path, branch_path])

    # Second block
    def split_input(input_tensor):
        return tf.split(input_tensor, num_or_size_splits=3, axis=3)

    split_layer = Lambda(split_input)(block_output)
    group1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    group3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    output_tensor = Concatenate()([group1, group2, group3])

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
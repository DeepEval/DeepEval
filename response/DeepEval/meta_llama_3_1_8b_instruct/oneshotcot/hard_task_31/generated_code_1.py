import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.layers import Dropout, Lambda
from keras import backend as K
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    drop_out = Dropout(0.2)(conv)

    def block(input_tensor):

        main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop_out)
        branch_path = input_tensor
        output_tensor = Add()([main_path, branch_path])

        return output_tensor
        
    block_output = block(input_tensor=input_layer)
    block_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block_output)

    def group_split(input_tensor):
        split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        group1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(split[0])
        group2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(split[1])
        group3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(split[2])
        return Concatenate()([group1, group2, group3])

    group_split_output = group_split(block_output)
    drop_out = Dropout(0.2)(group_split_output)

    flatten_layer = Flatten()(drop_out)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
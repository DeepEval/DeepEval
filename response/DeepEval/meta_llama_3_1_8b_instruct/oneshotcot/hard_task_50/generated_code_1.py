import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Concatenate, Lambda, Conv2D, SeparableConv2D
from keras.regularizers import l2
from tensorflow import split
import tensorflow as tf

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block1(input_tensor):

        path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        path1 = Flatten()(path1)
        
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        path2 = Flatten()(path2)
        
        path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        path3 = Flatten()(path3)

        output_tensor = Concatenate()([path1, path2, path3])
        output_tensor = Dropout(0.2)(output_tensor)

        return output_tensor

    block_output = block1(input_layer)
    block_output = Reshape(target_shape=(3,))(block_output)
    dense = Dense(units=128, activation='relu')(block_output)

    def block2(input_tensor):

        split_input = Lambda(lambda x: split(4, axis=-1)(x))(input_tensor)
        path1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
        path2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
        path3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
        path4 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split_input[3])
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    output_tensor = block2(Reshape(target_shape=(4, 1, 128))(dense))
    output_tensor = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(output_tensor)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
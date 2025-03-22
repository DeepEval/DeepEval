import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Flatten, Concatenate, Conv2D, Dense

def dl_model():

    input_layer = Input(shape=(32,32,3))
    
    def main_path(input_tensor):
        group_1 = input_tensor
        group_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        combined_group = Concatenate()([group_1, group_2, group_3])
        output_tensor = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)

        return output_tensor
    
    def branch_path(input_tensor):
        output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor

    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)

    combined_output = keras.layers.add([main_output, branch_output])
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
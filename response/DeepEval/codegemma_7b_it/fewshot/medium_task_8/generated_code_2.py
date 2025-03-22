import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Dense, Add, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        group1 = input_tensor
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        group3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor
    
    def branch_path(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv
    
    main_output = main_path(input_tensor=input_layer)
    branch_output = branch_path(input_tensor=input_layer)
    fused_output = Add()([main_output, branch_output])
    flatten = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
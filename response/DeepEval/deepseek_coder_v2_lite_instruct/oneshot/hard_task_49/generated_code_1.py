import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, Lambda, DepthwiseConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    concat_first_block = Concatenate()([flatten1, flatten2, flatten3])
    
    reshape_layer = Reshape((3, 1, 1))(concat_first_block)
    
    # Second block
    def split_and_process(tensor):
        return [
            DepthwiseConv2D(kernel_size=(1, 1), padding='same')(tensor[:, :, :, i])
            for i in range(4)
        ]
    
    split_layer = Lambda(split_and_process)(reshape_layer)
    
    concat_second_block = Concatenate()(split_layer)
    
    flatten_second_block = Flatten()(concat_second_block)
    
    dense_output = Dense(units=128, activation='relu')(flatten_second_block)
    output_layer = Dense(units=10, activation='softmax')(dense_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model
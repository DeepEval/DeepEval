import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, Concatenate, BatchNormalization, AveragePooling2D, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the first block
    def block1(input_tensor):
        path1 = DepthwiseConv2D(kernel_size=(1, 1), padding='same')(input_tensor)
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(path1)
        path2 = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path2)
        path3 = DepthwiseConv2D(kernel_size=(5, 5), padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same')(path3)
        outputs = Concatenate()([path1, path2, path3])
        outputs = BatchNormalization()(outputs)
        return outputs
    
    # Define the second block
    def block2(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path3)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(path3)
        path4 = AveragePooling2D(pool_size=(2, 2), padding='same')(input_tensor)
        outputs = Concatenate()([path1, path2, path3, path4])
        outputs = BatchNormalization()(outputs)
        return outputs
    
    # First block
    split_input = Lambda(lambda x: tf.split(x, 3, -1))(input_layer)
    output_block1 = block1(split_input[0])
    output_block1 = block1(split_input[1])
    output_block1 = block1(split_input[2])
    concat_output = Concatenate()([output_block1, output_block1, output_block1])
    concat_output = BatchNormalization()(concat_output)
    
    # Second block
    output_block2 = block2(concat_output)
    output_block2 = block2(concat_output)
    output_block2 = block2(concat_output)
    concat_output = Concatenate()([output_block2, output_block2, output_block2, concat_output])
    concat_output = BatchNormalization()(concat_output)
    
    flatten_layer = Flatten()(concat_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
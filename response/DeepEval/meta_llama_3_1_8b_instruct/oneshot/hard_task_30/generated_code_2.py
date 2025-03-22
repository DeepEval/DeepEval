import keras
from keras.layers import Input, Conv2D, Add, Lambda, Concatenate, BatchNormalization, DepthwiseConv2D, Reshape, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Define the main path of the first block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Define the branch path of the first block
    branch_path = input_layer
    
    # Combine the main and branch paths using addition
    block_output = Add()([main_path, branch_path])
    
    # Define the second block
    def block(input_tensor):
        inputs_split = Lambda(lambda x: tf.split(x, 3, axis=3))(input_tensor)
        
        # Define the depthwise separable convolutional layers for each group
        group1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same')(inputs_split[0])
        group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(group1)
        
        group2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs_split[1])
        group2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(group2)
        
        group3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs_split[2])
        group3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(group3)
        
        # Concatenate the outputs from the three groups
        output_tensor = Concatenate()([group1, group2, group3])
        
        return output_tensor
    
    block_output = block(block_output)
    bath_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
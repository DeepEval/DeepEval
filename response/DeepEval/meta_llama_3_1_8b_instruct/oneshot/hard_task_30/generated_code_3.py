import keras
from keras.layers import Input, Conv2D, Add, Lambda, DepthwiseConv2D, Reshape, Concatenate, BatchNormalization, Flatten, Dense
from keras import backend as K
import tensorflow as tf

def dl_model():
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    combined_output = Add()([main_path, branch_path])
    combined_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined_output)

    # Second block
    def depthwise_conv(x, kernel_size):
        x = DepthwiseConv2D(kernel_size=(kernel_size, kernel_size), strides=(1, 1), padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        return x
    
    def block(input_tensor):
        group1 = depthwise_conv(input_tensor, kernel_size=1)
        group2 = depthwise_conv(input_tensor, kernel_size=3)
        group3 = depthwise_conv(input_tensor, kernel_size=5)
        
        output_tensor = Concatenate()([group1, group2, group3])
        
        return output_tensor
        
    block_output = block(combined_output)
    bath_norm = BatchNormalization()(block_output)
    
    # Flatten the output
    flatten_layer = Flatten()(bath_norm)
    
    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
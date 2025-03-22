import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1: Conv block
    conv_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block)
    conv_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block)
    
    # Branch 2: Conv block
    conv_block_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_block_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block_2)
    conv_block_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block_2)
    
    # Branch 3: Conv block with separate main paths
    path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    
    # Combine the outputs of the three paths
    combined_path = Add()([path3, conv_block, conv_block_2])
    
    # Concatenate the outputs from the two blocks
    block_output = Concatenate()([combined_path, conv_block, conv_block_2])
    
    # Concatenate the outputs from the two blocks
    bath_norm = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(block_output)
    bath_norm = BatchNormalization()(bath_norm)
    
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
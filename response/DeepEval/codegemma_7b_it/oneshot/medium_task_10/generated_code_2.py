import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def basic_block(input_tensor):
        
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bath_norm = BatchNormalization()(conv)
        return bath_norm

    def residual_block(input_tensor):
        
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bath_norm = BatchNormalization()(conv)
        branch_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return bath_norm + branch_conv

    block_output = basic_block(input_tensor=conv)
    level1_output = residual_block(input_tensor=block_output)

    block_output = basic_block(input_tensor=level1_output)
    block_output = residual_block(input_tensor=block_output)
    level2_output = residual_block(input_tensor=block_output)

    global_branch_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv)
    level3_output = global_branch_conv + level2_output

    flatten_layer = Flatten()(level3_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
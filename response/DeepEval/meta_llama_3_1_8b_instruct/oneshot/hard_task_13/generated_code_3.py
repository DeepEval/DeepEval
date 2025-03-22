import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, BatchNormalization, Reshape, Dense, Multiply

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])
        return output_tensor
    
    block_output = block(input_layer)
    bath_norm = BatchNormalization()(block_output)
    
    def second_block(input_tensor):
        global_avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=64, activation='relu')(global_avg_pool)
        dense2 = Dense(units=64, activation='relu')(dense1)
        
        weights = Dense(units=64, activation='linear')(dense2)
        reshaped_weights = Reshape((1, 1, 64))(weights)
        
        element_wise_product = Multiply()([input_tensor, reshaped_weights])
        output_tensor = element_wise_product
        return output_tensor
    
    output_layer = second_block(bath_norm)
    final_dense = Dense(units=10, activation='softmax')(output_layer)
    
    model = keras.Model(inputs=input_layer, outputs=final_dense)
    
    return model
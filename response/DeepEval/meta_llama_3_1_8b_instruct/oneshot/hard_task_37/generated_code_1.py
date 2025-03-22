import keras
from keras.layers import Input, Conv2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    def block(input_tensor):
        
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        
        main_path1 = conv1
        main_path2 = conv2
        main_path3 = conv3
        
        parallel_branch = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        output_tensor = Concatenate()([Add()([main_path1, main_path2, main_path3]), parallel_branch])
        
        return output_tensor
    
    block_output1 = block(input_layer)
    block_output2 = block(block_output1)
    bath_norm = BatchNormalization()(block_output2)
    flatten_layer = Flatten()(bath_norm)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def block(input_tensor, filters, kernel_size, strides):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        return max_pooling
    
    main_path = block(input_layer, filters=32, kernel_size=(3, 3), strides=(1, 1))
    main_path = block(main_path, filters=64, kernel_size=(3, 3), strides=(2, 2))
    
    branch_path = block(input_layer, filters=64, kernel_size=(3, 3), strides=(2, 2))
    
    def addition_path(input_tensor1, input_tensor2):
        return Add()([input_tensor1, input_tensor2])
    
    combined_output = addition_path(main_path, branch_path)
    bath_norm = BatchNormalization()(combined_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
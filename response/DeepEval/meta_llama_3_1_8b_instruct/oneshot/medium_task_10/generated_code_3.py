import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = Activation('relu')(batch_norm)
    
    def basic_block(input_tensor):
        conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = Activation('relu')(batch_norm)
        conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        output = Add()([relu, conv_branch])
        return output
    
    res_block_1 = basic_block(relu)
    
    def res_block(input_tensor):
        conv_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        batch_norm_main = BatchNormalization()(conv_main)
        relu_main = Activation('relu')(batch_norm_main)
        
        conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(input_tensor)
        batch_norm_branch = BatchNormalization()(conv_branch)
        relu_branch = Activation('relu')(batch_norm_branch)
        conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu_branch)
        output = Add()([relu_main, conv_branch])
        return output
    
    res_block_2 = res_block(res_block_1)
    
    conv_global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(conv)
    batch_norm_global_branch = BatchNormalization()(conv_global_branch)
    relu_global_branch = Activation('relu')(batch_norm_global_branch)
    
    output = Add()([res_block_2, relu_global_branch])
    
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=(1, 1), padding='same')(output)
    flatten = Flatten()(avg_pool)
    dense = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model
import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Flatten, Dense, Concatenate
from keras import backend as K

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def block_1(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        gap_output = Dense(units=64, activation='relu')(gap)
        gap_output = Dense(units=10, activation='softmax')(gap_output)
        return gap_output
    
    def block_2(input_tensor):
        gmp = GlobalMaxPooling2D()(input_tensor)
        gmp_output = Dense(units=64, activation='relu')(gmp)
        gmp_output = Dense(units=10, activation='softmax')(gmp_output)
        return gmp_output
    
    gap_output = block_1(conv)
    gmp_output = block_2(conv)
    
    adding_layer = Add()([gap_output, gmp_output])
    channel_attn = Dense(units=10, activation='softmax')(adding_layer)
    channel_attn = Multiply()([channel_attn, conv])
    
    avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attn)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(channel_attn)
    
    concat = Concatenate()([avgpool, maxpool])
    
    channel_attn = Multiply()([conv, concat])
    channel_attn = Flatten()(channel_attn)
    
    output_layer = Dense(units=10, activation='softmax')(channel_attn)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
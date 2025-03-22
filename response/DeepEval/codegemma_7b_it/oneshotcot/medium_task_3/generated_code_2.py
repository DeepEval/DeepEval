import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = Conv2D(filters=20, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_tensor)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        return max_pooling

    block_output1 = block(input_tensor=input_layer)
    block_output2 = block(input_tensor=block_output1)
    
    concat = Concatenate()([block_output1, block_output2, input_layer])
    bath_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
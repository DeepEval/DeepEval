import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense, concatenate
from keras.regularizers import l2

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    conv_initial = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn = BatchNormalization()(conv)
        return bn
    
    block_output1 = block(conv_initial)
    block_output2 = block(block_output1)
    block_output3 = block(block_output2)

    added_output = Add()([conv_initial, block_output2, block_output3])

    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
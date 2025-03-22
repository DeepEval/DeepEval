import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    def convolution_block(input_tensor, num_filters, kernel_size):
        conv = Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv
    
    conv_block_output = convolution_block(input_layer, 32, (3, 3))
    conv_block_output = Concatenate()([input_layer, conv_block_output])
    conv_block_output = convolution_block(conv_block_output, 64, (3, 3))
    conv_block_output = Concatenate()([conv_block_output, input_layer])
    conv_block_output = convolution_block(conv_block_output, 128, (3, 3))

    bath_norm = BatchNormalization()(conv_block_output)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
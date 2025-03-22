import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense, Add

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def convolution_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = Concatenate()([input_tensor, conv1, conv2, conv3])
        return output_tensor
    
    conv_block_output = convolution_block(input_layer)
    add_layer = Add()([conv_block_output, conv_block_output])
    maxpool_layer = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(add_layer)
    
    flatten_layer = Flatten()(maxpool_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
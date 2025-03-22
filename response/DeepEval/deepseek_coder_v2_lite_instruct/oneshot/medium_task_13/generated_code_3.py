import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def conv_block(input_tensor, filters, kernel_size, strides):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(conv)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(conv)
        conv = BatchNormalization()(conv)
        return conv

    conv1 = conv_block(input_layer, filters=32, kernel_size=(3, 3), strides=(1, 1))
    conv2 = conv_block(conv1, filters=64, kernel_size=(3, 3), strides=(1, 1))
    conv3 = conv_block(conv2, filters=128, kernel_size=(3, 3), strides=(1, 1))
    
    concat_layer = Concatenate(axis=-1)([input_layer, conv1, conv2, conv3])
    
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model
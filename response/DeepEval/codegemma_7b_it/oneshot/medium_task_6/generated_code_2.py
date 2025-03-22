import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Lambda, add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    initial_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)

    def block(input_tensor):

        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        relu = Activation('relu')(batch_norm)
        return relu

    block1 = block(input_tensor=initial_conv)
    block2 = block(input_tensor=block1)
    block3 = block(input_tensor=block2)

    concat = add([initial_conv, block1, block2, block3])
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28,28,1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        drop1 = Dropout(0.25)(conv2)
        return drop1

    def block_2(input_tensor):
        add1 = Add()([block_1(input_tensor), input_tensor])
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(add1)
        drop2 = Dropout(0.5)(conv3)
        return drop2

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    output_layer = Flatten()(block2_output)
    dense = Dense(units=1024, activation='relu')(output_layer)
    output_layer = Dense(units=512, activation='relu')(dense)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
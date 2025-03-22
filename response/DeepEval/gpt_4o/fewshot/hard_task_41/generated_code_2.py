import keras
import tensorflow as tf
from keras.layers import Input, AveragePooling2D, Dropout, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        drop1 = Dropout(rate=0.5)(flatten1)

        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        drop2 = Dropout(rate=0.5)(flatten2)

        pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        drop3 = Dropout(rate=0.5)(flatten3)

        output_tensor = Concatenate()([drop1, drop2, drop3])
        return output_tensor

    def block_2(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense1 = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 8))(dense1)  # Assuming reshaping to (4, 4, 8) based on the units=128

    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense2 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
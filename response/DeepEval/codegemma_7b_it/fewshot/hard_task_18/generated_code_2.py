import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Reshape, GlobalAveragePooling2D

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return avg_pool

    def block_2(input_tensor):
        main_path = GlobalAveragePooling2D()(input_tensor)
        auxiliary_path = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        auxiliary_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(auxiliary_path)
        auxiliary_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(auxiliary_path)
        auxiliary_path = Flatten()(auxiliary_path)
        auxiliary_path = Dense(units=64, activation='relu')(auxiliary_path)
        auxiliary_path = Dense(units=128, activation='relu')(auxiliary_path)
        auxiliary_path = Reshape(target_shape=(1, 1, 128))(auxiliary_path)
        main_path = tf.keras.layers.multiply([main_path, auxiliary_path])
        return main_path

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
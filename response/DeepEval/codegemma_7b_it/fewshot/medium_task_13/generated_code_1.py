import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():

    input_layer = layers.Input(shape=(32, 32, 3))

    def conv_block(input_tensor, filters):
        conv = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        concat = layers.Concatenate(axis=-1)([conv, input_tensor])
        return concat

    block1_output = conv_block(input_tensor=input_layer, filters=32)
    block2_output = conv_block(input_tensor=block1_output, filters=64)
    block3_output = conv_block(input_tensor=block2_output, filters=128)

    flatten_layer = layers.Flatten()(block3_output)
    dense1 = layers.Dense(units=64, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
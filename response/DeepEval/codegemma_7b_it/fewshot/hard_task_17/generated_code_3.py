import keras
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Lambda, Flatten, Dense, Reshape, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        maxpool1 = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(maxpool1)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        return tf.reshape(dense2, [-1, input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[-1]])

    block1_output = block_1(input_tensor=input_layer)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    branch = block_1(input_tensor=maxpool)

    adding_layer = Add()([maxpool, branch])

    flatten = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Multiply
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    def block_2(input_tensor):
        avg_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
        reshaped = Reshape(target_shape=(input_tensor.shape[3],))(avg_pool)
        dense1 = Dense(units=64, activation='relu')(reshaped)
        dense2 = Dense(units=32, activation='relu')(dense1)
        weights = Dense(units=input_tensor.shape[3], activation='softmax')(dense2)
        weights = Reshape(target_shape=(1, 1, input_tensor.shape[3]))(weights)
        output_tensor = Multiply()([input_tensor, weights])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)

    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
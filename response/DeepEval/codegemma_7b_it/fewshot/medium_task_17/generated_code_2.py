import keras
import tensorflow as tf
from keras.layers import Input, Reshape, Lambda, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def reshape_groups(input_tensor):
        shape = tf.shape(input_tensor)
        input_tensor = tf.reshape(input_tensor, (shape[0], shape[1], shape[2], 3, shape[3] // 3))
        input_tensor = tf.transpose(input_tensor, perm=[0, 1, 2, 4, 3])
        input_tensor = tf.reshape(input_tensor, (shape[0], shape[1], shape[2], -1))
        return input_tensor

    reshaped_input = Lambda(reshape_groups)(input_layer)
    flatten_layer = Flatten()(reshaped_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
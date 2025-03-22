import keras
import tensorflow as tf
from keras.layers import Input, GlobalAveragePooling2D, Lambda, Flatten, Concatenate, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=input_tensor.shape[-1], activation='relu')(pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='sigmoid')(dense1)
        reshaped = Reshape((1, 1, input_tensor.shape[-1]))(dense2)
        output_tensor = Lambda(lambda x: tf.multiply(x[0], x[1]))([input_tensor, reshaped])
        return output_tensor

    branch_output = block(input_tensor)
    flatten = Flatten()(branch_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Reshape, Permute, Dense

def dl_model():

    input_layer = Input(shape=(None, None, 3))
    input_shape = keras.backend.int_shape(input_layer)
    input_reshaped = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(input_layer)
    input_transposed = Permute((0, 1, 3, 2))(input_reshaped)
    input_reshaped_back = Reshape((input_shape[1], input_shape[2], input_shape[3]))(input_transposed)

    dense_layer = Dense(units=10, activation='softmax')(input_reshaped_back)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model
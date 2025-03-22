import keras
from keras.layers import Input, Reshape, Permute, Conv2D, Dense, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Reshape input into groups
    input_shape = keras.backend.int_shape(input_layer)
    input_tensor = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(input_layer)

    # Swap third and fourth dimensions for channel shuffling
    input_tensor = Permute((0, 1, 3, 2))(input_tensor)

    # Reshape back to original input shape
    input_tensor = Reshape((input_shape[1], input_shape[2], input_shape[3]))(input_tensor)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(input_tensor)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
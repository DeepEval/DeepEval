import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    input_shape = keras.backend.int_shape(input_layer)
    
    # Reshape the input tensor into groups
    reshaped_input = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(input_layer)

    # Swap the third and fourth dimensions (channel shuffling)
    shuffled_input = Permute((2, 3, 1, 0))(reshaped_input)

    # Reshape back to the original input shape
    reshaped_output = Reshape((input_shape[1], input_shape[2], input_shape[3]))(shuffled_input)

    # Flatten and apply a fully connected layer
    flatten_layer = Flatten()(reshaped_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
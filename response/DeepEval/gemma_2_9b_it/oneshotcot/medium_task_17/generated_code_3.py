import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape and permute for channel shuffling
    input_shape = keras.backend.int_shape(input_layer)
    reshaped_input = Reshape((input_shape[1], input_shape[2], 3, input_shape[3] // 3))(input_layer)
    permuted_input = Permute((1, 2, 4, 3))(reshaped_input)
    reshaped_back = Reshape((input_shape[1], input_shape[2], input_shape[3]))(permuted_input)

    # Flatten and classify
    flattened = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
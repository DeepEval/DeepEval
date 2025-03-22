import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape input tensor
    input_shape = keras.backend.int_shape(input_layer)
    reshaped_input = Reshape((input_shape[1], input_shape[2], 3, 3))(input_layer) 

    # Permute dimensions
    permuted_input = Permute((2, 3, 1, 4))(reshaped_input) 

    # Reshape back to original shape
    reshaped_output = Reshape((input_shape[1], input_shape[2], 3))(permuted_input)

    # Flatten and classify
    flatten_layer = Flatten()(reshaped_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
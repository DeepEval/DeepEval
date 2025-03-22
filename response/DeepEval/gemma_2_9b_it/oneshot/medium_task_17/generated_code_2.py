import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input tensor into groups
    input_shape = keras.backend.int_shape(input_layer)
    reshaped_input = Reshape((input_shape[1], input_shape[2], 3, input_shape[3]//3))(input_layer)
    
    # Swap the third and fourth dimensions for channel shuffling
    permuted_input = Permute((1, 2, 4, 3))(reshaped_input) 
    
    # Reshape back to the original input shape
    reshaped_back = Reshape((input_shape[1], input_shape[2], input_shape[3]))(permuted_input)

    # Flatten and pass through a fully connected layer
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
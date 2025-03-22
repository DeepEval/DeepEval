import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3)) 

    # Reshape and shuffle channels
    input_shape = input_layer.shape[1:]
    groups = 3
    channels_per_group = input_shape[2] // groups
    reshaped_input = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(input_layer)
    permuted_input = Permute((2, 3, 1, 0))(reshaped_input)  
    reshaped_back = Reshape(input_shape)(permuted_input)

    # Fully connected layer for classification
    flatten = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
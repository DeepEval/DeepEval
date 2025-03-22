import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    input_shape = keras.backend.int_shape(input_layer)
    batch_size = input_shape[0]
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]
    groups = 3
    channels_per_group = channels // groups

    reshaped_input = Reshape((height, width, groups, channels_per_group))(input_layer)
    permuted_input = Permute((2, 3, 1, 0))(reshaped_input) 
    reshaped_back = Reshape((height, width, channels))(permuted_input)

    output_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
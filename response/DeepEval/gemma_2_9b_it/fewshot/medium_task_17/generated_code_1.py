import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Get the shape of the input tensor
    input_shape = keras.backend.shape(input_layer)

    # Reshape the input tensor into three groups
    groups = 3
    channels_per_group = input_shape[-1] // groups
    reshaped_input = Reshape((input_shape[1], input_shape[2], groups, channels_per_group))(input_layer)

    # Swap the third and fourth dimensions for channel shuffling
    permuted_input = Permute((2, 3, 1, 0))(reshaped_input)

    # Reshape back to the original input shape
    reshaped_output = Reshape((input_shape[1], input_shape[2], input_shape[3]))(permuted_input)

    # Flatten and pass through a fully connected layer with softmax activation
    flatten_layer = Flatten()(reshaped_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
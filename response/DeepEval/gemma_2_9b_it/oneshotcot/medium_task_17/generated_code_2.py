import keras
from keras.layers import Input, Reshape, Permute, Flatten, Dense
from keras.datasets import cifar10

def dl_model():  
    input_layer = Input(shape=(32, 32, 3))

    # Reshape the input tensor
    height, width, channels = input_layer.shape[1], input_layer.shape[2], input_layer.shape[3]
    reshaped_input = Reshape((height, width, 3, 1))(input_layer) 

    # Permute dimensions for channel shuffling
    permuted_input = Permute((2, 3, 1, 0))(reshaped_input)

    # Reshape back to original shape
    reshaped_back = Reshape((height, width, channels))(permuted_input)

    # Flatten and classify
    flatten_layer = Flatten()(reshaped_back)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
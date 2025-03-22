import keras
from keras.layers import Input, Reshape, Permute, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Reshape and channel shuffling
    reshape_input = Reshape((32, 32, 3, 1))(input_layer)
    permuted_input = Permute((0, 1, 3, 2))(reshape_input)

    # Reshape back to original shape
    reshape_back = Reshape((32, 32, 3))(permuted_input)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(reshape_back)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Reshape, Lambda, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # Reshape the input tensor into three groups
        reshape_tensor = Lambda(lambda x: keras.backend.reshape(x, (-1, x.shape[2], x.shape[3], 3)))(input_tensor)
        # Swap the third and fourth dimensions for channel shuffling
        permuted_tensor = Lambda(lambda x: keras.backend.permute_dimensions(x, (0, 2, 3, 1)))(reshape_tensor)
        # Reshape the tensor back to its original shape
        reshaped_tensor = Lambda(lambda x: keras.backend.reshape(x, (-1, x.shape[2] * x.shape[3])))(permuted_tensor)
        dense1 = Dense(units=128, activation='relu')(reshaped_tensor)
        return dense1

    def block_2(input_tensor):
        dense2 = Dense(units=64, activation='relu')(input_tensor)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    block1_output = block_1(input_tensor=input_layer)
    model = block_2(input_tensor=block1_output)

    return model
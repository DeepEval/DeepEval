import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Concatenate, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def same_block(input_tensor):
        pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=128, activation='relu')(pool)
        dense2 = Dense(units=input_tensor.shape[-1], activation='relu')(dense1)
        weights = Reshape(target_shape=input_tensor.shape[1:3] + [input_tensor.shape[-1]])(dense2)
        output_tensor = input_tensor * weights
        return output_tensor

    branch1_output = same_block(input_layer)
    branch2_output = same_block(input_layer)

    concatenated_output = Concatenate()([branch1_output, branch2_output])
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
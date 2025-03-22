import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten

 和 return model
def dl_model():
    input_shape = (32, 32, 3)

    branch_1 = GlobalAveragePooling2D()(input_shape)
    branch_2 = GlobalAveragePooling2D()(input_shape)

    same_block = Dense(units=32, activation='relu')(branch_1)
    same_block = Dense(units=32, activation='relu')(same_block)

    output = Concatenate()([branch_1, branch_2])
    flattened_output = Flatten()(output)

    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model
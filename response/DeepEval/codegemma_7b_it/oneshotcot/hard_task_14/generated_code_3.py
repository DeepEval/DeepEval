import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def main_path(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(gap)
        dense2 = Dense(units=input_tensor.shape[3], activation='sigmoid')(dense1)
        dense3 = Dense(units=input_tensor.shape[3], activation='sigmoid')(dense2)
        reshape = keras.layers.Reshape((input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))(dense3)
        output_tensor = keras.layers.Multiply()([reshape, input_tensor])
        return output_tensor

    def branch_path(input_tensor):
        conv = Conv2D(filters=input_tensor.shape[3], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_path_output = main_path(input_tensor)
    branch_path_output = branch_path(input_tensor)
    concat = keras.layers.add([main_path_output, branch_path_output])
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
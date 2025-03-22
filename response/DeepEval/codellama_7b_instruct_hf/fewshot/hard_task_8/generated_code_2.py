import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, Permute, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    primary_path = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(primary_path)
    primary_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(primary_path)

    branch_path = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)

    concatenated_path = keras.layers.concatenate([primary_path, branch_path], axis=3)

    reshaped_path = Permute((3, 1, 2))(concatenated_path)
    reshaped_path = Flatten()(reshaped_path)

    output_layer = Dense(units=10, activation='softmax')(reshaped_path)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
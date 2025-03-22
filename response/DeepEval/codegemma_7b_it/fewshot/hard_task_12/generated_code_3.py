import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 64))
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    def main_path(input_tensor):
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        output_tensor = Concatenate()([conv1_1, conv1_2])
        return output_tensor

    def branch_path(input_tensor):
        conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv2_1

    main_path_output = main_path(input_tensor=conv_1x1)
    branch_path_output = branch_path(input_tensor=conv_1x1)
    adding_layer = Add()([main_path_output, branch_path_output])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
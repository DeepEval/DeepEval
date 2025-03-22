import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def branch_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    def branch_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    def branch_3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    def branch_4(input_tensor):
        return input_tensor

    branch_1_output = branch_1(input_tensor=input_layer)
    branch_2_output = branch_2(input_tensor=input_layer)
    branch_3_output = branch_3(input_tensor=input_layer)
    branch_4_output = branch_4(input_tensor=input_layer)

    concat_output = Concatenate()([branch_1_output, branch_2_output, branch_3_output, branch_4_output])
    reshape_output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_output)

    main_path = Add()([input_layer, reshape_output])

    flatten_layer = Flatten()(main_path)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
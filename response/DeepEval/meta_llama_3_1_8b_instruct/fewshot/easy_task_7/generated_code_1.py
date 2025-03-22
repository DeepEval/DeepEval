import keras
from keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def main_path(input_tensor):
        conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv_block1 = Dropout(0.2)(conv_block1)
        conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block1)
        conv_block2 = Dropout(0.2)(conv_block2)
        conv_restore = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_block2)
        return conv_restore

    def branch_path(input_tensor):
        return input_tensor

    main_path_output = main_path(input_layer)
    branch_path_output = branch_path(input_layer)
    adding_layer = Add()([main_path_output, branch_path_output])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, DepthwiseConv2D, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def main_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=16, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        main_path_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)
        return main_path_output

    def branch_path(input_tensor):
        branch_path_output = Conv2D(filters=16, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return branch_path_output

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    concat_layer = Concatenate()([main_path_output, branch_path_output])
    add_layer = Add()([concat_layer, input_layer])
    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(units=32, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
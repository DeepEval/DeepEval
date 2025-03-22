import keras
from keras.layers import Input, DepthwiseConv2D, Lambda, BatchNormalization, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    def main_path(input_tensor):
        conv1 = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='valid', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    def branch_path(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1

    main_path_output = main_path(input_tensor=input_layer)
    branch_path_output = branch_path(input_tensor=input_layer)
    adding_layer = keras.layers.Add()([main_path_output, branch_path_output])

    flatten = Flatten()(adding_layer)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
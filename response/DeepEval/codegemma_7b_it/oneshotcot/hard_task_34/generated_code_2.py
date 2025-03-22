import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def main_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        separable_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', use_bias=False)(conv)
        bn = BatchNormalization()(separable_conv)
        return bn

    def branch_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    main_path_output = main_path(input_tensor)
    branch_path_output = branch_path(input_tensor)

    concat_features = Concatenate(axis=3)([main_path_output, branch_path_output])
    add_features = Multiply()([main_path_output, branch_path_output])

    flatten_layer = Flatten()(add_features)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model
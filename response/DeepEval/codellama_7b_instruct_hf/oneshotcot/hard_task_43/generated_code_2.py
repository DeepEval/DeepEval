import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense


def dl_model():
    input_layer = Input(shape=(28, 28, 1))


    def first_block(input_tensor):
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor


    first_block_output = first_block(input_tensor=input_layer)


    bath_norm = BatchNormalization()(first_block_output)


    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)


    dense2 = Dense(units=64, activation='relu')(dense1)


    def second_block(input_tensor):
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(branch1)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch2)
        branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(branch3)
        output_tensor = Concatenate()([branch1, branch2, branch3])
        return output_tensor

    second_block_output = second_block(input_tensor=dense2)


    output_layer = Dense(units=10, activation='softmax')(second_block_output)


    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, GlobalAveragePooling2D, Reshape, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool

    def block_2(input_tensor):
        main_path = GlobalAveragePooling2D()(input_tensor)
        main_path = Dense(units=64, activation='relu')(main_path)
        main_path = Dense(units=64, activation='relu')(main_path)

        reshaped = Reshape((1, 1, 64))(main_path)
        concat = Multiply()([reshaped, input_tensor])

        flatten = Flatten()(concat)
        output_layer = Dense(units=10, activation='softmax')(flatten)
        return output_layer

    block1_output = block_1(input_tensor=input_layer)
    block2_output = block_2(input_tensor=block1_output)
    final_output = Add()([block1_output, block2_output])

    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model
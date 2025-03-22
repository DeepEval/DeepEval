import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
        return maxpool

    path1_output = block_1(input_tensor=input_layer)

    def block_2(input_tensor):
        conv = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    path2_output = block_2(input_tensor=input_layer)

    adding_layer = Add()([path1_output, path2_output])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
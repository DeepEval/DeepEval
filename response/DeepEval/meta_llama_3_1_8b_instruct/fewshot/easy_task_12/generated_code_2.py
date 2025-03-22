import keras
from keras.layers import Input, Conv2D, MaxPooling2D, DepthwiseConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        relu = Dense(units=32, activation='relu')(input_tensor)
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation=None)(relu)
        conv = Dense(units=32, activation='relu')(conv)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
        return max_pool

    main_path = block(input_layer)
    main_path = block(main_path)

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)(input_layer)

    adding_layer = Add()([main_path, branch_path])
    adding_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(adding_layer)

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
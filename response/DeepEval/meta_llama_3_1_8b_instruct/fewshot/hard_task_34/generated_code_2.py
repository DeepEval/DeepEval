import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv)
        output_tensor = Concatenate()([conv, input_tensor])
        return output_tensor

    main_path = block(input_layer)
    for _ in range(2):
        main_path = block(main_path)

    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    adding_layer = Add()([main_path, branch_path])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
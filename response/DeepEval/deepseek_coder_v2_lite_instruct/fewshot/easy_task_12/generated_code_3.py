import keras
from keras.layers import Input, Conv2D, MaxPooling2D, SeparableConv2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor, filters):
        conv1 = SeparableConv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = SeparableConv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool

    main_path = block(input_layer, filters=32)
    main_path = block(main_path, filters=64)

    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    added_layer = Add()([main_path, branch_path])
    flatten_layer = Flatten()(added_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
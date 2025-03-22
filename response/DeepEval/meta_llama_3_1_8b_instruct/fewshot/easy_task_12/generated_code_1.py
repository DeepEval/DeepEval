import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthconv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
        max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(depthconv)
        return max_pooling

    main_path = block(input_layer)
    main_path = block(main_path)

    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path)

    adding_layer = Add()([main_path, branch_path])
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
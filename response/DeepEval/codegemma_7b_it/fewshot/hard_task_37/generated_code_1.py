import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        main_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv3)
        return main_path

    main_path_1 = block(input_tensor=input_layer)
    main_path_2 = block(input_tensor=input_layer)

    concat_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    concat_path = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(concat_path)

    adding_layer = Add()([main_path_1, main_path_2, concat_path])

    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
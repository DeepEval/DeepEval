import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, SeparableConv2D

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return maxpool

    main_path = block(input_tensor)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_tensor)
    concat_path = Add()([main_path, branch_path])

    flatten = Flatten()(concat_path)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
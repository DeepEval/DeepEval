import keras
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Concatenate, Flatten, Dense

def dl_model():
    def conv_block(x, filters):
        x = Conv2D(filters, (3, 3), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(filters, (3, 3), padding='same')(x)
        return x

    def conv_block_2(x, filters):
        x = Conv2D(filters, (5, 5), padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x = Conv2D(filters, (5, 5), padding='same')(x)
        return x

    input_layer = Input(shape=(28, 28, 1))

    x = conv_block(input_layer, 32)
    x = conv_block(x, 64)
    x = Flatten()(x)

    x = conv_block_2(input_layer, 64)
    x = conv_block_2(x, 128)
    x = Flatten()(x)

    x = Concatenate()([x, conv_block(input_layer, 128)])
    x = Conv2D(512, (5, 5), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
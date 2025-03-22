import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    conv_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool_block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_block1)

    conv_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool_block1)
    pool_block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_block2)

    # Combine outputs with input
    combined_output = Add()([pool_block2, input_layer])

    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
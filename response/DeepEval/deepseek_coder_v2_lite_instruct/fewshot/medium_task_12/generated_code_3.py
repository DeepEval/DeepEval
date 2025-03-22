import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters):
        conv = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv = BatchNormalization()(conv)
        conv = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(conv)
        conv = BatchNormalization()(conv)
        output_tensor = Add()([input_tensor, conv])
        return output_tensor

    # Block 1
    x = block(input_layer, 32)
    x = block(x, 32)
    x = block(x, 32)
    x = block(x, 32)

    # Block 2
    y = block(x, 64)
    y = block(y, 64)
    y = block(y, 64)
    y = block(y, 64)

    # Block 3
    z = block(y, 128)
    z = block(z, 128)
    z = block(z, 128)
    z = block(z, 128)

    # Flatten and fully connected layers
    flatten = Flatten()(z)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
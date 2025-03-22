import keras
from keras.layers import Input, BatchNormalization, Activation, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        for _ in range(3):
            bn = BatchNormalization()(input_tensor)
            act = Activation('relu')(bn)
            conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(act)
            input_tensor = Concatenate()([input_tensor, conv])
        return input_tensor

    path_one = block(input_tensor)
    path_two = block(input_tensor)

    merge = Concatenate()([path_one, path_two])
    flatten = Flatten()(merge)
    dense1 = Dense(units=256, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model
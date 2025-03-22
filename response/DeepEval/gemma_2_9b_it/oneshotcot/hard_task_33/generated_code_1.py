import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', depth_multiplier=1, activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
        return Add()([x, input_tensor])

    branch1 = block(input_layer)
    branch2 = block(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer))
    branch3 = block(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2))

    concatenated = Concatenate()([branch1, branch2, branch3])
    flatten = Flatten()(concatenated)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model
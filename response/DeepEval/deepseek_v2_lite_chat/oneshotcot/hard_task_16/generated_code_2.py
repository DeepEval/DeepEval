import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, multiply, add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    split = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)

    def block1(input_tensor):
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
        conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1_2)
        concat = Concatenate()([conv1_3, conv1_2, conv1_1])
        return concat

    def transition_conv(input_tensor):
        channels = keras.backend.int_shape(input_tensor)[-1]
        reshape = Reshape((-1, channels))(input_tensor)
        fully_connected1 = Dense(units=1024, activation='relu')(reshape)
        fully_connected2 = Dense(units=512, activation='relu')(fully_connected1)
        weights = Dense(units=channels)(fully_connected2)
        output_tensor = multiply([input_tensor, weights])
        return output_tensor

    def block2(input_tensor):
        global_pooling = MaxPooling2D(pool_size=(4, 4), strides=(1, 1), padding='valid')(input_tensor)
        fully_connected1 = Dense(units=1024, activation='relu')(global_pooling)
        fully_connected2 = Dense(units=512, activation='relu')(fully_connected1)
        output_tensor = Dense(units=10, activation='softmax')(fully_connected2)
        return output_tensor

    block1_output = block1(split[0])
    block2_output = block2(transition_conv(split[1]))
    model = add([block1_output, block2_output])

    return model
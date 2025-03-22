import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute


def dl_model():
    def block1(input_tensor):
        x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        return x

    def block2(input_tensor):
        x = Conv2D(32, (3, 3), activation='relu')(input_tensor)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (1, 1), activation='relu')(x)
        x = Permute((3, 1, 2))(x)
        x = Reshape((-1, 32))(x)
        return x


    input_layer = Input(shape=(28, 28, 1))
    block1_output = block1(input_layer)
    block2_output = block2(block1_output)

    x = Concatenate()([block1_output, block2_output])
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)


    return model
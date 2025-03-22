import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add, AveragePooling2D

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    def basic_block(input_tensor):
        conv_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bath_norm = BatchNormalization()(conv_path)
        output_tensor = Add()([input_tensor, bath_norm])
        return output_tensor

    block1 = basic_block(conv)
    block2 = basic_block(block1)

    branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)

    output = Add()([block2, branch])
    pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(output)
    flatten_layer = Flatten()(pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
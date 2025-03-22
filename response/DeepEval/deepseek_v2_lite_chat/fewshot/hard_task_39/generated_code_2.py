import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        maxpool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)
        flatten = Flatten()(maxpool2)
        return flatten

    def block_2(flatten_tensor):
        dense1 = Dense(units=128, activation='relu')(flatten_tensor)
        concat = Concatenate()([dense1, flatten_tensor])
        dense2 = Dense(units=64, activation='relu')(concat)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        model = keras.Model(inputs=input_layer, outputs=output_layer)
        return model

    flatten_tensor = block_1(input_tensor=input_layer)
    dl_model_output = block_2(flatten_tensor=flatten_tensor)

    return dl_model_output
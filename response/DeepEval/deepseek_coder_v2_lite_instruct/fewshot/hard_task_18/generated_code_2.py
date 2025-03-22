import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Dense, Flatten, Multiply, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return avg_pool

    block1_output = block_1(input_tensor=input_layer)
    main_path = block1_output

    added_path = Add()([input_layer, main_path])

    def block_2(input_tensor):
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=32, activation='relu')(gap)
        dense2 = Dense(units=32, activation='relu')(dense1)
        reshaped_weights = Dense(units=block1_output.shape[-1], activation='sigmoid')(dense2)
        reshaped_weights = keras.backend.expand_dims(reshaped_weights, axis=-1)
        reshaped_weights = keras.backend.expand_dims(reshaped_weights, axis=-1)
        weighted_input = Multiply()([input_tensor, reshaped_weights])
        return weighted_input

    block2_output = block_2(input_tensor=added_path)
    flattened_output = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
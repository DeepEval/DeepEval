import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Concatenate, Dense, Reshape, Flatten

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return pool

    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
        return pool

    main_path = block_1(input_layer)
    branch_path = block_2(input_layer)
    adding_layer = Add()([main_path, branch_path])

    def block_3(input_tensor):
        gap = keras.layers.GlobalAveragePooling2D()(input_tensor)
        weights = Dense(units=64, activation='relu')(gap)
        return weights

    block3_output = block_3(adding_layer)
    reshaped = Reshape(target_shape=(64,))(block3_output)
    weights = Dense(units=64, activation='relu')(reshaped)
    weighted_output = Multiply()([weights, reshaped])
    flatten = Flatten()(weighted_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
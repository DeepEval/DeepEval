import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    def block1(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

        return conv3

    def block2(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

        return conv2

    block1_output = block1(input_tensor)
    block2_output = block2(input_tensor)

    concat = Concatenate(axis=3)([block1_output, block2_output])
    bn = BatchNormalization()(concat)

    block1_shape = keras.backend.int_shape(block1_output)
    block2_shape = keras.backend.int_shape(block2_output)

    reshaped_block1 = Reshape((block1_shape[1], block1_shape[2], 1, block1_shape[3]))(block1_output)
    reshaped_block2 = Reshape((block2_shape[1], block2_shape[2], 1, block2_shape[3]))(block2_output)

    transposed_block2 = Permute((1, 2, 4, 3))(reshaped_block2)

    concat_shuffled = Concatenate(axis=3)([reshaped_block1, transposed_block2])

    reshaped_concat = Reshape((block1_shape[1], block1_shape[2], block1_shape[3] + block2_shape[3]))(concat_shuffled)

    flatten_layer = Flatten()(reshaped_concat)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    max_pooling_1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    max_pooling_2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    max_pooling_4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    max_pooling_1x1_flat = Flatten()(max_pooling_1x1)
    max_pooling_2x2_flat = Flatten()(max_pooling_2x2)
    max_pooling_4x4_flat = Flatten()(max_pooling_4x4)

    max_pooling_dropout = Dropout(0.2)(Concatenate()([max_pooling_1x1_flat, max_pooling_2x2_flat, max_pooling_4x4_flat]))

    # Block 2
    block_input = Reshape((1, 28, 28))(max_pooling_dropout)

    path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_input)

    path_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_input)
    path_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path_2)
    path_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path_2)

    path_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block_input)
    path_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path_3)
    path_3 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path_3)
    path_3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path_3)

    path_4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block_input)
    path_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path_4)

    path_4_flat = Flatten()(path_4)

    block_output = Concatenate()([path_1, path_2, path_3, path_4_flat])

    # Output layers
    block_output_flat = Flatten()(block_output)
    fc_layer_1 = Dense(units=64, activation='relu')(block_output_flat)
    fc_layer_2 = Dense(units=10, activation='softmax')(fc_layer_1)

    model = keras.Model(inputs=input_layer, outputs=fc_layer_2)

    return model
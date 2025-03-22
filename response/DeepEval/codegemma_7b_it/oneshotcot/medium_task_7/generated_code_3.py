import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    conv_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_1)

    conv_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_2)

    conv_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(pool_2)

    path_1 = conv_3
    path_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_3)
    path_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv_3)
    path_4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv_3)
    concat = Concatenate()([path_1, path_2, path_3, path_4])
    batch_norm = BatchNormalization()(concat)

    flatten = Flatten()(batch_norm)
    dense_1 = Dense(units=128, activation='relu')(flatten)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    output_layer = Dense(units=10, activation='softmax')(dense_2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    branch_1x1_3x3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1x1_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch_1x1_3x3)

    branch_max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    concat = Concatenate()([branch_3x3, branch_1x1_3x3, branch_max_pool])
    batch_norm = BatchNormalization()(concat)

    flatten_layer = Flatten()(batch_norm)

    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)

    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
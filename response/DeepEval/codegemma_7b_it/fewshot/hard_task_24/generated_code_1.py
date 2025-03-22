import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    initial_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(initial_conv)

    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(initial_conv)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = UpSampling2D(size=(2, 2))(branch_2)

    branch_3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(initial_conv)
    branch_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3)
    branch_3 = UpSampling2D(size=(4, 4))(branch_3)

    fusion = Concatenate()([branch_1, branch_2, branch_3])

    fusion_conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(fusion)

    flatten_layer = Flatten()(fusion_conv)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    branch_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)

    branch_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=128, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=128, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)

    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    concat = Concatenate()([branch_1, branch_2, branch_3])

    dropout = Dropout(0.5)(concat)
    flatten_layer = Flatten()(dropout)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
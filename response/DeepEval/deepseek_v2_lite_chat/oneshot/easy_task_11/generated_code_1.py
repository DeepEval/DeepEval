import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.regularizers import l2

def dl_model():

    input_layer = Input(shape=(28, 28, 1))
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool = AveragePooling2D(pool_size=(5, 5), strides=3)(conv)

    def block(input_tensor):

        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.001))(input_tensor)
        flatten = Flatten()(conv)

        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        dropout = Dropout(0.5)(dense2)

        output_layer = Dense(units=10, activation='softmax')(dropout)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model
        
    model = block(input_tensor=pool)

    return model
import keras
from keras.layers import Input, Conv2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    split_layer = Lambda(lambda x: K.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])
    output_tensor = Concatenate()([path1, path2, path3])

    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
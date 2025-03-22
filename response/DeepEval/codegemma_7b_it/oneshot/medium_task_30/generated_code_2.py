import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    concat_pool = Concatenate()([avg_pool_1, avg_pool_2, avg_pool_3])
    flatten_layer = Flatten()(concat_pool)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
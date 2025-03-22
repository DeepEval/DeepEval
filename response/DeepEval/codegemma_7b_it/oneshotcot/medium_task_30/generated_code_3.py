import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    avg_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    avg_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    concat_output = Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    flatten_output = Flatten()(concat_output)

    dense1 = Dense(units=64, activation='relu')(flatten_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model
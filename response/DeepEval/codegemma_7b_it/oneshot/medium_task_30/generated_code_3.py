import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)

    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)

    concat_output = Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    flatten_output = Flatten()(concat_output)

    dense1 = Dense(units=128, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
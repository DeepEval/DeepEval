import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    avg_pooling_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    avg_pooling_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    avg_pooling_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)

    concat_pooling = Concatenate()([avg_pooling_1, avg_pooling_2, avg_pooling_3])

    flatten_layer = Flatten()(concat_pooling)

    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
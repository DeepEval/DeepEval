import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(conv)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)

    concat_features = Concatenate()([avg_pool_1, avg_pool_2, avg_pool_3])
    flatten_layer = Flatten()(concat_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
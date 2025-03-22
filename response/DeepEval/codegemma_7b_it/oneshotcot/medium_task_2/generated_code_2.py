import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():    
    
    input_layer = Input(shape=(32, 32, 3))
    path_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    path_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(path_main)
    path_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(path_main)
    path_branch = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    output_tensor = Concatenate()([path_main, path_branch])
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
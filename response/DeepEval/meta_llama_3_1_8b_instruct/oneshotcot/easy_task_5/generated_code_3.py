import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    reduce_dim = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    feature_extraction = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reduce_dim)
    restore_dim = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(feature_extraction)
    flatten_layer = Flatten()(restore_dim)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
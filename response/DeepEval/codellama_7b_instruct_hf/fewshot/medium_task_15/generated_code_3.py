import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import ReLU
from keras.layers.normalization import GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv)
    norm = BatchNormalization()(pool)
    relu = ReLU()(norm)
    flatten = Flatten()(relu)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    input_reshape = Flatten()(input_layer)
    input_reshape = Reshape(target_shape=(16,))(input_reshape)
    dense3 = Dense(units=64, activation='relu')(input_reshape)
    dense4 = Dense(units=10, activation='softmax')(dense3)

    cat = Concatenate()([dense4, dense2])
    output_layer = Dense(units=10, activation='softmax')(cat)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
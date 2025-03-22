import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Multiply, GlobalAveragePooling2D, Dense, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)

    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    global_avg_pool = GlobalAveragePooling2D()(max_pooling2)

    weights = Dense(units=64, activation='relu')(global_avg_pool)
    weights = Reshape((1, 1, 64))(weights)

    multiplied = Multiply()([max_pooling2, weights])
    multiplied = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(multiplied)
    multiplied = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(multiplied)

    flatten_layer = Flatten()(multiplied)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
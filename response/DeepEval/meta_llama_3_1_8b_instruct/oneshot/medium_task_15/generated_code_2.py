import keras
from keras.layers import Input, Conv2D, BatchNormalization, GlobalAveragePooling2D, Reshape, Multiply, Concatenate, Conv2DTranspose, AveragePooling2D, Dense
from keras.models import Model

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)

    global_avg_pool = GlobalAveragePooling2D()(batch_norm)

    flatten_layer = Reshape((32,))(global_avg_pool)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)

    reshape_layer = Reshape((32, 32))(dense2)
    multiply_layer = Multiply()([reshape_layer, batch_norm])
    concat_layer = Concatenate()([input_layer, multiply_layer])

    downsample = Conv2DTranspose(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')(concat_layer)
    downsample = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(downsample)
    output_layer = Dense(units=10, activation='softmax')(downsample)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
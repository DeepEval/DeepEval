import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(conv)
    relu = ReLU()(batch_norm)

    global_avg_pool = GlobalAveragePooling2D()(relu)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    reshaped = Reshape(target_shape=(32, 32, 64))(dense2)
    weighted_feature_maps = Multiply()([reshaped, relu])
    concatenated = Concatenate()([input_layer, weighted_feature_maps])

    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv_1x1)
    output_layer = Dense(units=10, activation='softmax')(avg_pool)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
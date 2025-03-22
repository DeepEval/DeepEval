import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Reshape, Concatenate, AveragePooling2D

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 
    conv_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn_layer = BatchNormalization()(conv_layer)
    
    pool_layer = GlobalAveragePooling2D()(bn_layer)
    
    dense1 = Dense(units=32, activation='relu')(pool_layer)
    dense2 = Dense(units=32, activation='relu')(dense1)

    reshape_layer = Reshape((32, 32, 32))(dense2)
    weighted_features = reshape_layer * bn_layer

    concat_layer = Concatenate()([input_layer, weighted_features])
    
    downsample_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_layer)
    downsample_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(downsample_conv)

    output_layer = Dense(units=10, activation='softmax')(downsample_pool)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))
    conv_increase_dim = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_feature_extract = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_uniform')(conv_increase_dim)
    conv_reduce_dim = Conv2D(filters=8, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(conv_feature_extract)

    flatten_layer = Flatten()(conv_reduce_dim)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model
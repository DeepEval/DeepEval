import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, DepthwiseConv2D, Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv_dw = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    bn_dw = BatchNormalization()(conv_dw)
    conv_pw = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn_dw)
    bn_pw = BatchNormalization()(conv_pw)

    conv_dw1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(bn_pw)
    bn_dw1 = BatchNormalization()(conv_dw1)
    conv_pw1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn_dw1)
    bn_pw1 = BatchNormalization()(conv_pw1)

    # Branch Path
    conv_br = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)

    # Combine Paths
    combined = Add()([bn_pw1, conv_br])

    # Classification Layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model
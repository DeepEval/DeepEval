import keras
from keras.layers import Input, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    depthwise_separable_conv = DepthwiseSeparableConv2D(
        filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu'
    )(input_layer)
    batch_norm = BatchNormalization()(depthwise_separable_conv)
    flatten = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    concatenate = keras.layers.concatenate([input_layer, dense2])
    output_layer = Dense(units=10, activation='softmax')(concatenate)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model
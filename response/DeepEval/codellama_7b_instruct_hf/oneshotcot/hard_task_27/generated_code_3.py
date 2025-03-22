import keras
from keras.layers import Input, DepthwiseSeparableConv2D, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    depthwise_separable_conv = DepthwiseSeparableConv2D(kernel_size=(7, 7), padding='same', activation='relu')(input_layer)
    batch_norm = BatchNormalization()(depthwise_separable_conv)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model